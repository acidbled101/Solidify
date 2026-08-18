"""
Append-only JSONL trace format for a DPO Inspector run.

One event per line:

    {"seq": 12, "t": 1785600000.42, "type": "grad_step", "payload": {...}}

Why JSONL, append-only: a live SSE stream and a full replay are the exact
same read (open the file, read lines) -- there is no separate "live" and
"saved" code path. A run that crashes mid-write still leaves every event up
to the crash fully readable; TraceReader treats a truncated/corrupt final
line as "not there yet", not an error, so a live tailer racing the writer
never crashes on a half-flushed line.

TraceWriter is also where the rich Python objects dpo_branch.py's on_event
callback hands over (trimesh.Trimesh, geometric_judge.JudgeScore/
JudgeDetails, numpy arrays/scalars, dataclasses in general) get turned into
JSON-safe payloads:

  - trimesh.Trimesh -> exported to <run_dir>/<seq>_<n>.glb; the payload gets
    a small summary dict (file name, vertex/face count, watertight, bounds)
    instead of the mesh itself. Meshes are NEVER inlined into the trace --
    a 1M-face mesh has no business going through JSON/SSE.
  - Any numpy array -> reduced to a histogram (counts/edges/n/min/max/mean),
    never sent raw, same reasoning -- geometric_judge.JudgeDetails' per-face/
    per-vertex/per-ray arrays can be 10^5-10^6 elements on an undecimated
    mesh (score_mesh's own max_faces_for_scoring default is 20000, but
    JudgeDetails.faces_scored can still be that large).
  - Any dataclass (JudgeScore, JudgeDetails, JudgeWeights, DPOBranchReport,
    ...) -> recursed field-by-field. This is deliberately generic rather
    than one hand-written converter per type: JudgeDetails' array fields
    fall through to the numpy-array branch automatically, so adding a field
    to any dataclass upstream does not require touching this file.
  - numpy scalar types -> native Python via .item()-equivalent casts.
  - Anything else unrecognized -> str(value), so a trace write can never
    hard-fail generation over a serialization gap; check the trace for
    stray stringified objects if something looks wrong.
"""
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


def _histogram(arr, bins: int = 40) -> dict:
    """Bin a 1D array into counts+edges. Never returns raw values."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "counts": [], "edges": [], "min": None, "max": None, "mean": None}
    if arr.min() == arr.max():
        # np.histogram needs a non-degenerate range; a single-valued array
        # (e.g. every face passes at the exact same angle) is real data, not
        # an error -- report it as a single bin rather than dropping it.
        return {
            "n": int(arr.size), "counts": [int(arr.size)],
            "edges": [float(arr.min()), float(arr.min())],
            "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
        }
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "n": int(arr.size),
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


class TraceWriter:
    """Appends JSON-safe events to <run_dir>/trace.jsonl, exporting any
    trimesh.Trimesh in a payload to its own GLB file alongside it."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.run_dir / "trace.jsonl"
        self._fh = open(self._path, "a", buffering=1)  # line-buffered: readers see events promptly
        self._seq = 0
        self._mesh_seq = 0
        # Resume seq numbering if this writer reopens an existing trace
        # (shouldn't normally happen -- one writer per run -- but a resumed
        # process after a crash should not silently restart seq at 0 and
        # produce duplicate/ambiguous sequence numbers in the same file).
        if self._path.exists():
            for evt in TraceReader(self.run_dir).read_all():
                self._seq = max(self._seq, evt.get("seq", 0) + 1)

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> dict:
        safe_payload = self._to_jsonsafe(payload or {})
        event = {"seq": self._seq, "t": time.time(), "type": event_type, "payload": safe_payload}
        self._fh.write(json.dumps(event) + "\n")
        self._seq += 1
        return event

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- serialization -------------------------------------------------

    def _to_jsonsafe(self, value):
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            v = float(value)
            return v if math.isfinite(v) else None
        if isinstance(value, np.ndarray):
            return _histogram(value)

        # Imported lazily -- trace.py must be importable (e.g. by
        # server.py/runner.py's argument parsing) without torch/trimesh/
        # trellis2 already on sys.path.
        import trimesh
        if isinstance(value, trimesh.Trimesh):
            return self._export_mesh(value)

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self._to_jsonsafe(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonsafe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._to_jsonsafe(v) for k, v in value.items()}

        # Anything else (a live callable, a torch tensor that slipped
        # through, ...): stringify rather than raise. A trace write must
        # never be the thing that crashes a 20-minute generation run.
        return str(value)

    def _export_mesh(self, mesh) -> dict:
        self._mesh_seq += 1
        base = f"{self._seq:04d}_{self._mesh_seq}"
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)

        glb_name, stl_name, export_error = None, None, None
        if n_faces > 0:
            # GLB is what viewer.js loads for in-browser rendering; STL is
            # exported alongside so a caller can download it straight into an
            # external mesh-diff tool or slicer -- no reason to make every
            # consumer pick one export format up front. Independent try/except
            # per format: a GLB writer quirk shouldn't cost the STL too.
            try:
                mesh.export(str(self.run_dir / f"{base}.glb"))
                glb_name = f"{base}.glb"
            except Exception as e:
                export_error = str(e)
            try:
                mesh.export(str(self.run_dir / f"{base}.stl"))
                stl_name = f"{base}.stl"
            except Exception as e:
                export_error = export_error or str(e)

        result = {
            "file": glb_name,
            "stl_file": stl_name,
            "vertex_count": int(n_verts),
            "face_count": int(n_faces),
            "watertight": bool(mesh.is_watertight) if n_faces > 0 else None,
            "bounds": mesh.bounds.tolist() if n_verts > 0 else None,
        }
        if export_error:
            result["export_error"] = export_error
        return result


class TraceReader:
    """Reads a trace.jsonl. Tolerant of a truncated final line (a writer
    still mid-flush, or a crashed run) -- that line is simply not returned
    yet, not raised as an error."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "trace.jsonl"

    def read_all(self) -> list:
        return list(self.iter_from(0))

    def iter_from(self, byte_offset: int = 0) -> Iterator[dict]:
        """Yield every complete JSON line at/after byte_offset. Does not
        track a new offset for the caller -- see tail_from for that."""
        if not self.path.exists():
            return
        with open(self.path, "r") as fh:
            fh.seek(byte_offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Either a genuinely corrupt line, or (far more likely)
                    # we read a partial line the writer hadn't finished
                    # flushing. Either way: stop, don't raise. A live tailer
                    # will pick it up whole on the next poll once the writer
                    # finishes the line.
                    return

    def tail_from(self, byte_offset: int = 0):
        """Like iter_from, but also returns the byte offset to resume from
        next time -- the position right after the last COMPLETE line read,
        never mid-line. Returns (events: list[dict], next_offset: int)."""
        if not self.path.exists():
            return [], byte_offset
        events = []
        with open(self.path, "r") as fh:
            fh.seek(byte_offset)
            pos = byte_offset
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    pos += len(line)
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    break  # partial line -- stop before advancing pos past it
                pos += len(line)
        return events, pos

    def is_finished(self) -> bool:
        """A trace is 'finished' once the RUNNER's own top-level
        session_end/error event has been written -- used by the server to
        know when it can stop tailing.

        Deliberately NOT dpo_branch.py's "run_end" event: that only marks
        the shape-SLat sampling stage done. Print-prep (repair/diagnostics/
        export, run by runner.py after sampling returns) still runs after
        it and can still fail -- checking for dpo_branch's "run_end" here
        would make the server think a run finished while post-processing
        was still in flight."""
        for evt in self.read_all():
            if evt.get("type") in ("session_end", "error"):
                return True
        return False
