"""
FastAPI application: upload -> queue -> poll -> preview/download.

`import trellis_core` is the very first trellis-related import (before anything
that transitively pulls in torch), so the MPS-fallback env vars and sys.path
setup run before torch is loaded.
"""

# MUST be first: sets env vars + sys.path before torch is imported anywhere.
import trellis_core  # noqa: F401,E402

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from . import config, worker
from .jobs import JobStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trellis.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    worker.start_worker(store)
    worker.warm_pipeline_async()
    log.info("Server started. Model=%s device=%s", config.MODEL_ID, config.DEVICE)
    yield


app = FastAPI(title="TRELLIS.2 image-to-3D", lifespan=lifespan)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def _parse_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_bool(value) -> bool:
    if value is None:
        return False
    return str(value).lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    seed: str = Form(None),
    pipeline_type: str = Form(None),
    texture_size: str = Form(None),
    target_faces: str = Form(None),
    skip_printable: str = Form(None),
):
    # Read + size-limit the upload.
    image_bytes = await image.read()
    if not image_bytes:
        return _error(400, "Empty upload.")
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        return _error(413, f"Image exceeds the {config.MAX_UPLOAD_MB} MB limit.")

    # Validate it is a real image (verify() decodes headers without full load).
    try:
        import io

        PILImage.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return _error(400, "Uploaded file is not a valid image.")

    # Validate + normalize params.
    try:
        seed_v = _parse_int(seed, config.DEFAULT_SEED)
        target_faces_v = _parse_int(target_faces, config.DEFAULT_TARGET_FACES)
        texture_size_v = _parse_int(texture_size, config.DEFAULT_TEXTURE_SIZE)
    except ValueError:
        return _error(400, "seed, target_faces and texture_size must be integers.")

    pipeline_type_v = pipeline_type or config.DEFAULT_PIPELINE_TYPE
    if pipeline_type_v not in config.ALLOWED_PIPELINE_TYPES:
        return _error(400, f"pipeline_type must be one of {config.ALLOWED_PIPELINE_TYPES}.")
    if texture_size_v not in config.ALLOWED_TEXTURE_SIZES:
        return _error(400, f"texture_size must be one of {config.ALLOWED_TEXTURE_SIZES}.")
    if target_faces_v <= 0:
        return _error(400, "target_faces must be a positive integer.")

    skip_printable_v = (
        _as_bool(skip_printable) if skip_printable is not None else config.SKIP_PRINTABLE_BY_DEFAULT
    )

    params = {
        "seed": seed_v,
        "pipeline_type": pipeline_type_v,
        "texture_size": texture_size_v,
        "target_faces": target_faces_v,
        "skip_printable": skip_printable_v,
    }

    job = store.create(params, image_bytes, image.filename or "upload")
    worker.job_queue.put(job.id)
    return JSONResponse(status_code=202, content={"job_id": job.id, "status": "queued"})


@app.get("/api/jobs")
async def list_jobs():
    jobs = store.list_recent(20)
    return {"jobs": [j.to_public_dict(debug=config.DEBUG) for j in jobs]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = store.get(job_id)
    except KeyError:
        return _error(404, "Unknown job id.")
    return job.to_public_dict(debug=config.DEBUG)


@app.get("/api/jobs/{job_id}/files/{filename}")
async def get_job_file(job_id: str, filename: str):
    try:
        job = store.get(job_id)
    except KeyError:
        return _error(404, "Unknown job id.")

    # Build an allowlist of exact known basenames from the job's result. Never
    # join a user-supplied filename onto a path without confirming it is one of
    # these -- prevents path traversal.
    known = set()
    if job.result and job.result.get("files"):
        known = {f["filename"] for f in job.result["files"]}
    if filename not in known:
        return _error(404, "Unknown file for this job.")

    path = os.path.join(job.output_dir, filename)
    if not os.path.exists(path):
        return _error(404, "File not found on disk.")
    return FileResponse(path, filename=filename)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline_loaded": worker.is_pipeline_loaded(),
        "queue_depth": worker.queue_depth(),
    }


@app.get("/api/config")
async def get_config():
    return {
        "pipeline_types": list(config.ALLOWED_PIPELINE_TYPES),
        "texture_sizes": list(config.ALLOWED_TEXTURE_SIZES),
        "default_pipeline_type": config.DEFAULT_PIPELINE_TYPE,
        "default_texture_size": config.DEFAULT_TEXTURE_SIZE,
        "default_seed": config.DEFAULT_SEED,
        "default_target_faces": config.DEFAULT_TARGET_FACES,
        "skip_printable_by_default": config.SKIP_PRINTABLE_BY_DEFAULT,
        "no_texture": config.NO_TEXTURE,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "model_id": config.MODEL_ID,
    }


# --------------------------------------------------------------------------
# Static frontend (mounted last so it doesn't shadow /api/*).
# --------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
