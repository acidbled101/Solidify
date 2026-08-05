"""Live training dashboard -- a READER of run directories, never a supervisor.

Deliberately has no ability to start, stop, or hold the trainer. It tails
`metrics.jsonl`, reads `status.json`, and writes `control.json`. That is the
whole interface. Killing this server, or having it crash, cannot affect a run
in progress -- which is the point, because it is the component most likely to
be restarted while a weekend run is mid-flight.

Binds 0.0.0.0 so a phone on the same Tailscale network can reach it at the
Mac's tailnet IP. There is no auth: this is intended to sit behind Tailscale,
which does the authentication. Do NOT expose the port to the public internet.

Run:
    python -m devlab.train_server                  # :8200, all interfaces
    python -m devlab.train_server --port 8201
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trellis_core import train_run as tr  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RUNS_DIR = os.environ.get("TRELLIS_RUNS_DIR", str(REPO / "runs"))

app = FastAPI(title="TRELLIS training dashboard")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def api_runs():
    return {"runs": tr.list_runs(RUNS_DIR), "runs_dir": RUNS_DIR}


def _run_dir(run_id: str) -> str:
    d = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(d):
        raise HTTPException(404, f"no such run: {run_id}")
    return d


@app.get("/api/runs/{run_id}/metrics")
def api_metrics(run_id: str, offset: int = 0):
    """Incremental tail. The client passes back the `offset` it last received,
    so a dashboard left open for a weekend transfers only new records rather
    than re-downloading a metrics file that has grown to tens of megabytes."""
    d = _run_dir(run_id)
    records, new_offset = tr.read_metrics(os.path.join(d, "metrics.jsonl"), offset)
    return {"records": records, "offset": new_offset}


@app.get("/api/runs/{run_id}/status")
def api_status(run_id: str):
    d = _run_dir(run_id)
    out: Dict[str, Any] = {}
    for name in ("status", "meta", "control"):
        try:
            with open(os.path.join(d, f"{name}.json")) as f:
                out[name] = json.load(f)
        except Exception:
            out[name] = None
    st = out.get("status") or {}
    hb = st.get("heartbeat")
    out["alive"] = bool(hb and (time.time() - hb) < 180)
    out["stale_s"] = (time.time() - hb) if hb else None
    return out


class ControlPatch(BaseModel):
    lr: Optional[float] = None
    pause: Optional[bool] = None
    stop: Optional[bool] = None
    eval_now: Optional[bool] = None
    eval_every: Optional[int] = None
    checkpoint_every: Optional[int] = None
    log_every: Optional[int] = None
    sample_gallery: Optional[bool] = None
    note: Optional[str] = None


@app.post("/api/runs/{run_id}/control")
def api_control(run_id: str, patch: ControlPatch):
    """Merge a patch into control.json and bump `version`.

    The version bump is what makes the trainer apply it -- without it the
    trainer sees an unchanged file and keeps its current settings. Merging
    rather than replacing means a phone sending `{"lr": 5e-5}` doesn't
    silently reset every other knob to its default.
    """
    d = _run_dir(run_id)
    current = tr.Control.read(d)
    fields = {k: v for k, v in patch.model_dump().items() if v is not None}
    for k, v in fields.items():
        setattr(current, k, v)
    current.version = int(current.version) + 1
    current.write(d)
    return {"ok": True, "control": current.__dict__}


@app.get("/api/runs/{run_id}/samples")
def api_samples(run_id: str):
    """Index of eval sample artifacts, newest checkpoint first."""
    d = _run_dir(run_id)
    root = os.path.join(d, "samples")
    out = []
    if os.path.isdir(root):
        for step_dir in sorted(os.listdir(root), reverse=True):
            p = os.path.join(root, step_dir)
            if not os.path.isdir(p):
                continue
            files = sorted(os.listdir(p))
            out.append({
                "step": int(step_dir.replace("step_", "")) if step_dir.startswith("step_") else None,
                "dir": step_dir,
                "images": [f for f in files if f.lower().endswith((".png", ".jpg", ".webp"))],
                "meshes": [f for f in files if f.lower().endswith((".glb", ".stl", ".ply"))],
            })
    return {"checkpoints": out}


@app.get("/api/runs/{run_id}/samples/{step_dir}/{filename}")
def api_sample_file(run_id: str, step_dir: str, filename: str):
    d = _run_dir(run_id)
    # Contain the path: these components arrive from a URL, and a dashboard
    # bound to 0.0.0.0 must not be able to serve arbitrary files off the disk.
    for part in (step_dir, filename):
        if os.sep in part or part in ("..", ".") or part.startswith("."):
            raise HTTPException(400, "bad path")
    path = os.path.realpath(os.path.join(d, "samples", step_dir, filename))
    if not path.startswith(os.path.realpath(os.path.join(d, "samples")) + os.sep):
        raise HTTPException(400, "bad path")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.get("/api/runs/{run_id}/log")
def api_log(run_id: str, tail: int = 400):
    d = _run_dir(run_id)
    p = os.path.join(d, "train.log")
    if not os.path.exists(p):
        return {"lines": []}
    with open(p, "rb") as f:
        # Seek from the end rather than reading the file: an overnight run's
        # log can reach hundreds of MB and the dashboard polls this.
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        data = f.read().decode("utf-8", "replace")
    return {"lines": data.splitlines()[-tail:]}


@app.get("/api/host")
def api_host():
    """Addresses this dashboard can be reached at, including the tailnet IP.

    Surfaced in the UI so the phone URL is copy-pasteable rather than
    something you have to go and look up on the Mac while away from it.
    """
    info: Dict[str, Any] = {"hostname": socket.gethostname()}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["lan_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["lan_ip"] = None
    try:
        r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
        info["tailscale_ip"] = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        info["tailscale_ip"] = None
    return info


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(STATIC / "train.html")


@app.get("/train.js")
def train_js():
    return FileResponse(STATIC / "train.js")


@app.get("/train.css")
def train_css():
    return FileResponse(STATIC / "train.css")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args(argv)

    global RUNS_DIR
    if args.runs_dir:
        RUNS_DIR = args.runs_dir

    import uvicorn
    host_info = api_host()
    print(f"runs dir : {RUNS_DIR}")
    print(f"local    : http://127.0.0.1:{args.port}")
    if host_info.get("lan_ip"):
        print(f"lan      : http://{host_info['lan_ip']}:{args.port}")
    if host_info.get("tailscale_ip"):
        print(f"tailscale: http://{host_info['tailscale_ip']}:{args.port}   <- use this from your phone")
    else:
        print("tailscale: not detected (run `tailscale ip -4` to check)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
