"""Run directory, metrics stream, live control channel, and checkpoint manager.

THE CONTRACT THIS ENFORCES
--------------------------
Training must survive everything that is not training. The dashboard may crash,
the terminal may close, the assistant's context may run out, the laptop lid may
shut -- none of it can be allowed to interrupt or corrupt a weekend-long run.

That rules out any design where the trainer talks to a server, holds a socket,
or depends on a supervising process. Instead every channel here is a **file**:

    runs/<run_id>/
        meta.json       written once: config, dataset fingerprint, git sha
        metrics.jsonl   APPEND-ONLY stream of training/eval/system records
        status.json     heartbeat, atomically replaced (never partially read)
        control.json    read BY the trainer, written by anyone else
        ckpt/step_*/    resumable checkpoints
        samples/step_*/ decoded eval meshes + renders
        train.log       raw stdout/stderr of the detached process

Readers (the dashboard, me, you from a phone) only ever tail `metrics.jsonl`
and replace `control.json`. Neither can block or crash the trainer:
`read_control` swallows every exception, because a half-written control file
during a weekend run must degrade to "keep going with the previous settings",
never to a traceback in the training loop.

Append-only + line-delimited is what makes the metrics stream safe to read
while it is being written: a reader that catches a torn final line just drops
it and picks it up on the next poll.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional

RUNS_DIR_DEFAULT = "runs"


# ---------------------------------------------------------------------------
# Control channel
# ---------------------------------------------------------------------------


@dataclass
class Control:
    """Live-editable knobs. The trainer re-reads this file every step.

    `version` is the apply trigger: the trainer only re-applies settings when
    it *changes*, so an unchanged file costs one stat + parse per step and
    never fights the training loop for ownership of the learning rate. Bump it
    on every edit or the edit is ignored.

    Only fields that are safe to change mid-run live here. Anything that would
    invalidate the optimizer state or the dataset (model, LoRA rank, batch
    size) is deliberately absent -- those require a restart, and a restart
    resumes from the last checkpoint anyway.
    """

    version: int = 0
    lr: Optional[float] = None          # None = leave as configured
    pause: bool = False                 # hold at the top of the loop, keep process alive
    stop: bool = False                  # checkpoint, then exit cleanly
    eval_now: bool = False              # force an eval at the next step boundary
    eval_every: Optional[int] = None
    checkpoint_every: Optional[int] = None
    log_every: Optional[int] = None
    sample_gallery: Optional[bool] = None
    note: str = ""                      # free text, surfaced on the dashboard

    @staticmethod
    def path(run_dir: str) -> str:
        return os.path.join(run_dir, "control.json")

    @classmethod
    def read(cls, run_dir: str, fallback: Optional["Control"] = None) -> "Control":
        """Never raises. A malformed or half-written file yields `fallback`.

        This is called from inside the training loop, so correctness here is
        less important than never throwing: losing one edit is recoverable,
        killing an 18-hour run is not.
        """
        try:
            with open(cls.path(run_dir)) as f:
                data = json.load(f)
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except Exception:
            return fallback if fallback is not None else cls()

    def write(self, run_dir: str) -> None:
        atomic_write_json(self.path(run_dir), asdict(self))


# ---------------------------------------------------------------------------
# Atomic status
# ---------------------------------------------------------------------------


def atomic_write_json(path: str, obj: Any) -> None:
    """Write via temp file + rename so a reader never sees a partial document.

    `os.replace` is atomic within a filesystem, which is what lets the
    dashboard poll `status.json` at any frequency without ever catching a
    truncated write.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def _git_sha(repo: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class TrainRun:
    """Owns one run directory. Safe to construct twice on the same id (resume)."""

    def __init__(self, run_id: str, runs_dir: str = RUNS_DIR_DEFAULT, repo: Optional[str] = None):
        self.run_id = run_id
        self.dir = os.path.join(runs_dir, run_id)
        self.repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for sub in ("", "ckpt", "samples"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self._metrics_fh = None
        self._t0 = time.time()

    # -- metrics ----------------------------------------------------------

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.dir, "metrics.jsonl")

    def _fh(self):
        if self._metrics_fh is None:
            self._metrics_fh = open(self.metrics_path, "a", buffering=1)  # line buffered
        return self._metrics_fh

    def log(self, kind: str, step: int, **fields) -> None:
        """Append one record. `kind` is 'train' | 'eval' | 'system' | 'event'.

        Line-buffered and flushed per record: a reader tailing the file sees
        each metric within milliseconds, and a hard kill loses at most the
        record being written rather than the buffer behind it.
        """
        rec = {"t": time.time(), "step": int(step), "kind": kind, **fields}
        fh = self._fh()
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()

    # -- status heartbeat -------------------------------------------------

    def status(self, **fields) -> None:
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "heartbeat": time.time(),
            "elapsed_s": time.time() - self._t0,
            **fields,
        }
        atomic_write_json(os.path.join(self.dir, "status.json"), payload)

    # -- meta -------------------------------------------------------------

    def write_meta(self, config: Dict[str, Any]) -> None:
        atomic_write_json(os.path.join(self.dir, "meta.json"), {
            "run_id": self.run_id,
            "started_at": time.time(),
            "git_sha": _git_sha(self.repo),
            "config": config,
        })

    # -- checkpoints ------------------------------------------------------

    def ckpt_dir(self, step: int) -> str:
        d = os.path.join(self.dir, "ckpt", f"step_{step:08d}")
        os.makedirs(d, exist_ok=True)
        return d

    def sample_dir(self, step: int) -> str:
        d = os.path.join(self.dir, "samples", f"step_{step:08d}")
        os.makedirs(d, exist_ok=True)
        return d

    def latest_checkpoint(self) -> Optional[str]:
        """Most recent checkpoint that finished writing.

        A checkpoint counts as complete only when it contains `DONE`. A run
        killed mid-save leaves a directory without that marker, and resuming
        from it would load half a state dict -- so it is skipped and the
        previous one is used instead.
        """
        root = os.path.join(self.dir, "ckpt")
        if not os.path.isdir(root):
            return None
        cands = sorted(
            d for d in os.listdir(root)
            if d.startswith("step_") and os.path.exists(os.path.join(root, d, "DONE"))
        )
        return os.path.join(root, cands[-1]) if cands else None

    @staticmethod
    def mark_checkpoint_done(ckpt_dir: str) -> None:
        with open(os.path.join(ckpt_dir, "DONE"), "w") as f:
            f.write(str(time.time()))

    def prune_checkpoints(self, keep: int = 3) -> None:
        """Keep the newest `keep` complete checkpoints. LoRA adapters are small,
        but eval samples and optimizer state are not, and a weekend run at one
        checkpoint per 500 steps would otherwise fill the disk unattended."""
        root = os.path.join(self.dir, "ckpt")
        done = sorted(
            d for d in os.listdir(root)
            if d.startswith("step_") and os.path.exists(os.path.join(root, d, "DONE"))
        )
        import shutil
        for d in done[:-keep] if keep > 0 else []:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)


# ---------------------------------------------------------------------------
# Reader side (dashboard)
# ---------------------------------------------------------------------------


def read_metrics(path: str, offset: int = 0) -> tuple:
    """Read records from `offset` bytes. Returns (records, new_offset).

    Tolerates a torn trailing line: the file is being appended to while we
    read, so the last line may be incomplete. We only advance `offset` past
    lines that parsed, which means the partial line is simply re-read (whole)
    on the next poll.
    """
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records, offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()

    # Consume only up to the final newline. Anything after it is either a
    # partial write or nothing, and in both cases must be left for the next
    # poll. Doing this by rfind rather than by catching a parse failure is
    # what keeps the byte accounting exact: `split(b"\n")` appends a trailing
    # empty element after the final newline, and counting a byte for it
    # advances `offset` one too far, which decapitates the first line of the
    # next read so that it never parses and is silently lost. That was a real
    # bug here -- over a weekend it drops one record per poll.
    end = data.rfind(b"\n")
    if end < 0:
        return records, offset

    consumed = 0
    for raw in data[:end + 1].split(b"\n")[:-1]:
        if raw.strip():
            try:
                records.append(json.loads(raw))
            except Exception:
                # A complete line that will not parse is corruption, not a
                # tear (tears cannot reach here). Skip it rather than stalling
                # the offset forever on an unattended run.
                pass
        consumed += len(raw) + 1
    return records, offset + consumed


def list_runs(runs_dir: str = RUNS_DIR_DEFAULT) -> List[Dict[str, Any]]:
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for rid in sorted(os.listdir(runs_dir), reverse=True):
        d = os.path.join(runs_dir, rid)
        if not os.path.isdir(d):
            continue
        entry = {"run_id": rid, "dir": d}
        for name in ("status", "meta"):
            try:
                with open(os.path.join(d, f"{name}.json")) as f:
                    entry[name] = json.load(f)
            except Exception:
                entry[name] = None
        st = entry.get("status") or {}
        hb = st.get("heartbeat")
        # 180s: comfortably longer than the slowest expected step (~76s) plus
        # an eval, so a busy trainer is never mislabelled dead.
        entry["alive"] = bool(hb and (time.time() - hb) < 180)
        out.append(entry)
    return out
