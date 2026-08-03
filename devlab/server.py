"""
DPO Inspector -- a standalone, no-auth, dev-only web app for watching and
analyzing the physics-aware DPO branching step (trellis_core/dpo_branch.py).

Deliberately separate from server/ (the production SOLIDIFY app): different
port, no shared auth, no shared job queue. A DPO run needs ~19GB unified
memory and cannot coexist with the warm production pipeline (measured --
see report/dpo_feasibility.tex) -- see prod_server_status() below, which is
exactly the check that would have caught the memory-contention/auth-outage
incident from the session that built this tool, had it existed then.

Run:
    python -m devlab.server            # http://127.0.0.1:8100

Every run is a subprocess (devlab/runner.py), never a thread: a hard process
boundary means an OOM-kill or crash during a 20+ minute branch step can't
take this server down with it, and gives us a real exit code + log tail to
show instead of a wedged thread with no diagnosis.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from devlab.trace import TraceReader

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVLAB_DIR = Path(__file__).resolve().parent
TRACES_DIR = DEVLAB_DIR / "traces"
STATIC_DIR = DEVLAB_DIR / "static"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

PROD_LAUNCHD_LABEL = "com.trellis.webserver"
PROD_PORT = 8000
MESH_FILENAME_RE = re.compile(r"^[0-9]{4}_[0-9]+\.(glb|stl)$")


# ---------------------------------------------------------------------------
# Production-server contention guard
# ---------------------------------------------------------------------------
# This is the check that would have caught, before it happened, the incident
# from the session that built this tool: starting a memory-hungry DPO run
# while the production server was warm led to swap exhaustion, and
# separately, switching the working-tree branch while the launchd-managed
# prod server auto-restarted briefly served it WITHOUT auth. Both are
# consequences of the same underlying fact -- this repo's production
# service and its dev tooling share one machine's resources and one git
# checkout's working tree -- so surfacing it loudly, every time, is cheap
# insurance against repeating either mistake.

def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def _launchd_loaded(label: str) -> bool:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=3)
        return label in out.stdout
    except Exception:
        return False


def prod_server_status() -> dict:
    port_up = _port_open(PROD_PORT)
    launchd_up = _launchd_loaded(PROD_LAUNCHD_LABEL)
    warning = None
    if port_up or launchd_up:
        warning = (
            "The production server appears to be running "
            + ("(port 8000 reachable" + (", launchd job loaded)" if launchd_up else ")")
               if port_up else "(launchd job loaded, port not yet up)")
            + ". A DPO run needs ~19GB of unified memory and measurably "
              "contended with the warm production pipeline in testing -- "
              "stop it first:\n"
              f"  launchctl bootout gui/$(id -u)/{PROD_LAUNCHD_LABEL}\n"
              "Restore it afterwards with:\n"
              f"  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{PROD_LAUNCHD_LABEL}.plist"
        )
    return {"prod_port_open": port_up, "prod_launchd_loaded": launchd_up, "prod_warning": warning}


# ---------------------------------------------------------------------------
# Run management -- one subprocess at a time, matching the plan's own memory
# analysis (two DPO-scale processes would never fit either).
# ---------------------------------------------------------------------------

class RunManager:
    def __init__(self):
        self.active: dict[str, subprocess.Popen] = {}

    def _reap(self) -> None:
        for run_id in list(self.active):
            if self.active[run_id].poll() is not None:
                del self.active[run_id]

    def is_busy(self) -> bool:
        self._reap()
        return len(self.active) > 0

    def active_run_id(self) -> Optional[str]:
        self._reap()
        return next(iter(self.active), None)

    def start(self, run_id: str, extra_args: list) -> subprocess.Popen:
        self._reap()
        if self.active:
            raise RuntimeError(f"run {next(iter(self.active))!r} is already in progress")
        run_dir = TRACES_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "devlab.runner", "--run-dir", str(run_dir), *extra_args]
        log_fh = open(run_dir / "runner.log", "wb")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_fh, stderr=subprocess.STDOUT)
        self.active[run_id] = proc
        return proc

    def cancel(self, run_id: str) -> bool:
        self._reap()
        proc = self.active.get(run_id)
        if proc is None:
            return False
        proc.terminate()
        return True


runs = RunManager()
app = FastAPI(title="DPO Inspector")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    status = prod_server_status()
    return {"ok": True, "busy": runs.is_busy(), "active_run_id": runs.active_run_id(), **status}


@app.post("/api/run")
async def start_run(
    image: UploadFile = File(...),
    pipeline_type: str = Form("512"),
    steps: str = Form(""),
    seed: str = Form("42"),
    target_faces: str = Form("1000000"),
    t_branch: str = Form("0.5"),
    num_branches: str = Form("1"),
    branch_noise_scale: str = Form("0.02"),
    continuation_steps: str = Form("2"),
    num_delta_grad_steps: str = Form("3"),
    dpo_beta: str = Form("1.0"),
    vanilla_too: str = Form(""),
    acknowledge_prod_warning: str = Form(""),
):
    if runs.is_busy():
        raise HTTPException(409, f"run {runs.active_run_id()!r} is already in progress")

    status = prod_server_status()
    ack = acknowledge_prod_warning.strip().lower() in ("1", "true", "on", "yes")
    if status["prod_warning"] and not ack:
        # 409, not 400: this is a real precondition-failed, and the
        # frontend's job is to show the banner and let the user explicitly
        # retry with acknowledge_prod_warning=1 -- never silently proceed.
        raise HTTPException(409, status["prod_warning"])

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image upload")
    if pipeline_type not in ("512", "1024"):
        raise HTTPException(400, "pipeline_type must be '512' or '1024' (dpo_branch.py doesn't support 1024_cascade)")

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = TRACES_DIR / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(image.filename or "")[1] or ".png"
    image_path = input_dir / f"image{ext}"
    image_path.write_bytes(data)

    args = [
        "--image", str(image_path),
        "--pipeline-type", pipeline_type,
        "--seed", seed,
        "--target-faces", target_faces,
        "--t-branch", t_branch,
        "--num-branches", num_branches,
        "--branch-noise-scale", branch_noise_scale,
        "--continuation-steps", continuation_steps,
        "--num-delta-grad-steps", num_delta_grad_steps,
        "--dpo-beta", dpo_beta,
    ]
    if steps.strip():
        args += ["--steps", steps.strip()]
    if vanilla_too.strip().lower() in ("1", "true", "on", "yes"):
        args += ["--vanilla-too"]

    try:
        runs.start(run_id, args)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    return {"run_id": run_id}


@app.post("/api/cancel/{run_id}")
def cancel_run(run_id: str):
    if not runs.cancel(run_id):
        raise HTTPException(404, "run not active (already finished, or unknown id)")
    return {"cancelled": True}


def _pid_alive(pid: Optional[int]) -> bool:
    """Best-effort liveness check via signal 0 (send-nothing, just probe).
    False for None/0/missing-process; True if the process exists, even if
    it's owned by another user and we couldn't actually signal it."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


@app.get("/api/runs")
def list_runs():
    out = []
    if not TRACES_DIR.exists():
        return {"runs": out}
    for d in sorted(TRACES_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        events = TraceReader(d).read_all()
        if not events:
            continue
        start = next((e for e in events if e["type"] == "session_start"), None)
        end = next((e for e in reversed(events) if e["type"] in ("session_end", "error")), None)
        start_payload = (start or {}).get("payload", {})

        finished = end is not None
        ok = (end or {}).get("type") == "session_end"
        error = (end or {}).get("payload", {}).get("message") if (end or {}).get("type") == "error" else None

        # A run with no session_end/error AND no longer tracked by THIS
        # server process AND whose OS process is actually gone is not
        # "running" -- it's orphaned, most commonly because the dev server
        # itself got restarted (e.g. to deploy a code change) while a
        # generation was still in flight. Left undetected this shows as a
        # permanently-stuck "running…" sidebar entry with no way to tell it
        # apart from a real one.
        if not finished and d.name not in runs.active and not _pid_alive(start_payload.get("pid")):
            finished = True
            ok = False
            error = (
                f"interrupted -- process pid={start_payload.get('pid')} is no longer running "
                "and never wrote a final event (most likely the dev server was restarted "
                "while this run was still in progress)."
            )

        out.append({
            "run_id": d.name,
            "started_at": events[0]["t"],
            "image": start_payload.get("image"),
            "pipeline_type": start_payload.get("pipeline_type"),
            "seed": start_payload.get("seed"),
            "finished": finished,
            "ok": ok,
            "error": error,
            "duration_seconds": (events[-1]["t"] - events[0]["t"]) if len(events) > 1 else 0.0,
            "n_events": len(events),
        })
    return {"runs": out}


@app.get("/api/trace/{run_id}")
def get_trace(run_id: str):
    run_dir = TRACES_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run_id")
    return {"run_id": run_id, "events": TraceReader(run_dir).read_all()}


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str):
    """SSE: replays every event already on disk, then tails new ones as
    runner.py writes them, until session_end/error. A finished run's
    replay and a live run's tail are the exact same code path -- the only
    difference is how many iterations of the poll loop return nothing new
    before is_finished() becomes true."""
    import asyncio

    run_dir = TRACES_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run_id")

    async def gen():
        reader = TraceReader(run_dir)
        offset = 0
        while True:
            events, offset = reader.tail_from(offset)
            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
            if reader.is_finished() and not events:
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/mesh/{run_id}/{filename}")
def get_mesh(run_id: str, filename: str):
    run_dir = TRACES_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run_id")
    # Only ever serve filenames matching exactly what TraceWriter._export_mesh
    # produces (SEQ_N.glb / SEQ_N.stl) -- never join client input onto a
    # filesystem path without validating its shape first (path traversal),
    # same pattern as server/main.py's get_job_file allowlist check.
    if not MESH_FILENAME_RE.match(filename):
        raise HTTPException(400, "invalid mesh filename")
    path = run_dir / filename
    if not path.is_file():
        raise HTTPException(404, "mesh not found")
    media = "model/gltf-binary" if filename.endswith(".glb") else "model/stl"
    return FileResponse(str(path), media_type=media)


@app.get("/api/output/{run_id}/{filename}")
def get_output_file(run_id: str, filename: str):
    """Serves the final printable GLB/STL, or the vanilla-comparison GLB/OBJ
    (runner.py writes both under <run_dir>/output/ when --vanilla-too is
    set -- run_generation() has no STL export path, only glb+obj), same
    filename-shape validation as get_mesh."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(glb|stl|obj)", filename):
        raise HTTPException(400, "invalid output filename")
    path = TRACES_DIR / run_id / "output" / filename
    if not path.is_file():
        raise HTTPException(404, "file not found")
    media = {"glb": "model/gltf-binary", "stl": "model/stl", "obj": "text/plain"}[filename.rsplit(".", 1)[-1]]
    return FileResponse(str(path), media_type=media)


@app.get("/api/compare")
def compare_runs(ids: str):
    """Aligned summary metrics across 2+ runs -- for the n>1 question the
    feasibility report's step 4 needs (does steering win more than chance
    across several images), once enough runs exist to compare."""
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    out = []
    for rid in run_ids:
        run_dir = TRACES_DIR / rid
        if not run_dir.is_dir():
            continue
        events = TraceReader(run_dir).read_all()
        by_type: dict = {}
        for e in events:
            by_type.setdefault(e["type"], []).append(e["payload"])
        branch_point = (by_type.get("branch_point") or [{}])[0]
        printable_result = (by_type.get("printable_result") or [{}])[0]
        grad_steps = by_type.get("grad_step") or []
        run_end_events = by_type.get("run_end") or []
        dpo_report = (run_end_events[-1] or {}).get("report") if run_end_events else None
        out.append({
            "run_id": rid,
            "branch_point": branch_point,
            "printable_result": printable_result,
            "grad_step_count": len(grad_steps),
            "loss_history": [g.get("loss") for g in grad_steps],
            "dpo_report": dpo_report,
        })
    return {"runs": out}


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# Mounted at /static (not /) to match every /static/... reference in
# index.html/app.js/viewer.js/charts.js -- mounting at "/" would shadow the
# /api/* routes above entirely (StaticFiles claims every path under its
# mount, including ones that don't exist on disk, when html=True) and also
# wouldn't match any of this frontend's own asset paths.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    status = prod_server_status()
    print("=" * 70)
    print("DPO Inspector -- dev-only, no auth, localhost")
    if status["prod_warning"]:
        print()
        print("WARNING:")
        for line in status["prod_warning"].splitlines():
            print(f"  {line}")
        print()
    print("  http://127.0.0.1:8100")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")


if __name__ == "__main__":
    main()
