"""
FastAPI application: upload -> queue -> poll -> preview/download.

`import trellis_core` is the very first trellis-related import (before anything
that transitively pulls in torch), so the MPS-fallback env vars and sys.path
setup run before torch is loaded.
"""

# MUST be first: sets env vars + sys.path before torch is imported anywhere.
import trellis_core  # noqa: F401,E402

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from starlette.middleware.base import BaseHTTPMiddleware

from . import config, db, worker
from .jobs import JobStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trellis.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    # Optional one-time convenience: seed the first operator from env vars if the
    # allow-list is still empty (handy for an unattended launchd deploy). After
    # this, manage operators with `python -m server.admin`.
    if config.AUTH_SEED_USER and config.AUTH_SEED_PASSWORD and db.count_operators() == 0:
        db.add_operator(config.AUTH_SEED_USER, config.AUTH_SEED_PASSWORD)
        log.info("Seeded initial operator '%s' from environment.", config.AUTH_SEED_USER)

    worker.start_worker(store)
    worker.warm_pipeline_async()

    n_ops = db.count_operators()
    if n_ops:
        log.info(
            "Server started. Model=%s device=%s auth=ON (%d operator(s))",
            config.MODEL_ID, config.DEVICE, n_ops,
        )
    else:
        log.warning(
            "Server started with NO OPERATORS -> auth is OPEN (bootstrap mode). "
            "Fine on a trusted LAN, but before exposing this beyond your own network "
            "run `python -m server.admin add <name>` to create the first login; auth "
            "turns on automatically the moment one operator exists."
        )
    yield


app = FastAPI(title="TRELLIS.2 image-to-3D", lifespan=lifespan)


# --------------------------------------------------------------------------
# Session tokens (stateless, HMAC-signed) -- used for "stay signed in".
# Token = base64url(username|expiry) + "." + base64url(HMAC-SHA256(secret, ...)).
# The secret is a stable per-install value persisted in the DB, so tokens keep
# working across server restarts.
# --------------------------------------------------------------------------
def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_session_token(username: str, ttl_days: int) -> str:
    exp = int(time.time()) + ttl_days * 86400
    payload = f"{username}|{exp}".encode("utf-8")
    sig = hmac.new(db.get_or_create_session_secret().encode("utf-8"), payload, hashlib.sha256).digest()
    return _b64u(payload) + "." + _b64u(sig)


def verify_session_token(token: str) -> str | None:
    """Return the username if the token's signature is valid and unexpired."""
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _b64u_decode(payload_b64)
        expected = hmac.new(
            db.get_or_create_session_secret().encode("utf-8"), payload, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64u_decode(sig_b64), expected):
            return None
        username, exp = payload.decode("utf-8").rsplit("|", 1)
        if int(exp) < time.time():
            return None
        return username
    except Exception:
        return None


def _authenticate(request: Request) -> str | None:
    """Resolve the operator from a Bearer session token or Basic credentials."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        user = verify_session_token(auth[7:].strip())
        # A still-valid token for a since-removed operator must stop working.
        if user and db.operator_exists(user):
            return user
        return None
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, _, password = decoded.partition(":")
        except Exception:
            return None
        if db.verify_operator(user, password):
            return user.strip()
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """Real, enforced login -- gated on the operators table in the local DB.

    While no operators exist the app runs OPEN (bootstrap mode) so the machine
    can never be locked out of itself; the moment `python -m server.admin add`
    creates the first operator, every `/api/*` route requires a valid credential.

    The static shell (`/`, `/static/*`) and `POST /api/login` are always
    reachable so the branded SOLIDIFY login page can load and sign in -- that
    page is the real gate, not the browser's native Basic-Auth popup."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Static shell + any non-API route: always served (login page lives here).
        if path == "/" or path == "/static" or path.startswith("/static/") or not path.startswith("/api/"):
            return await call_next(request)
        # The login endpoint checks credentials itself and must stay reachable.
        if path == "/api/login":
            return await call_next(request)
        # No operators yet -> open bootstrap mode.
        if db.count_operators() == 0:
            request.state.user = "anonymous"
            return await call_next(request)
        user = _authenticate(request)
        if not user:
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="SOLIDIFY"'},
                content="Authentication required.",
            )
        request.state.user = user
        return await call_next(request)


app.add_middleware(AuthMiddleware)


@app.middleware("http")
async def _revalidate_frontend(request: Request, call_next):
    """Force browsers to revalidate the frontend against its ETag on every load.

    Starlette's StaticFiles (and the index FileResponse) send ETag/Last-Modified
    but no Cache-Control, so mobile browsers cache the shell heuristically and can
    keep running a stale app.js/index.html after an update -- the classic "works
    on desktop, broken/old on my phone" symptom. `no-cache` means "you may store
    it, but you must revalidate before reuse"; unchanged files still return a
    cheap 304, changed files are picked up immediately."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


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
@app.post("/api/login")
async def login(request: Request):
    """Verify operator credentials (sent as Basic auth, as the frontend builds)
    and return a signed session token for "stay signed in"."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Basic "):
        return _error(401, "Missing credentials.")
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        user, _, password = decoded.partition(":")
    except Exception:
        return _error(401, "Malformed credentials.")
    user = user.strip()
    if not db.verify_operator(user, password):
        return JSONResponse(status_code=401, content={"error": "Invalid operator ID or access key."})
    token = make_session_token(user, config.SESSION_TTL_DAYS)
    return {"token": token, "username": user}


@app.post("/api/jobs")
async def create_job(
    request: Request,
    image: UploadFile = File(...),
    seed: str = Form(None),
    pipeline_type: str = Form(None),
    texture_size: str = Form(None),
    target_faces: str = Form(None),
    skip_printable: str = Form(None),
    skip_texture: str = Form(None),
):
    if worker.queue_depth() >= config.MAX_QUEUE_DEPTH:
        return _error(429, f"Queue is full ({config.MAX_QUEUE_DEPTH} jobs waiting). Try again shortly.")

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
    skip_texture_v = (
        _as_bool(skip_texture) if skip_texture is not None else config.SKIP_TEXTURE_BY_DEFAULT
    )

    params = {
        "seed": seed_v,
        "pipeline_type": pipeline_type_v,
        "texture_size": texture_size_v,
        "target_faces": target_faces_v,
        "skip_printable": skip_printable_v,
        "skip_texture": skip_texture_v,
    }

    operator = getattr(request.state, "user", "anonymous")
    job = store.create(params, image_bytes, image.filename or "upload", operator=operator)
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
    avg = db.avg_job_seconds()
    return {
        "pipeline_types": list(config.ALLOWED_PIPELINE_TYPES),
        "texture_sizes": list(config.ALLOWED_TEXTURE_SIZES),
        "default_pipeline_type": config.DEFAULT_PIPELINE_TYPE,
        "default_texture_size": config.DEFAULT_TEXTURE_SIZE,
        "default_seed": config.DEFAULT_SEED,
        "default_target_faces": config.DEFAULT_TARGET_FACES,
        "skip_printable_by_default": config.SKIP_PRINTABLE_BY_DEFAULT,
        "skip_texture_by_default": config.SKIP_TEXTURE_BY_DEFAULT,
        "no_texture": config.NO_TEXTURE,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "model_id": config.MODEL_ID,
        # Live per-job time estimate: real average of recent jobs, else a default.
        "avg_job_seconds": avg if avg is not None else config.DEFAULT_JOB_MINUTES * 60,
    }


# --------------------------------------------------------------------------
# Library -- the shared, persistent archive of every completed generation.
# --------------------------------------------------------------------------
@app.get("/api/library")
async def library():
    """Every generated model, newest first, shared across all operators."""
    return {"models": db.list_models()}


@app.post("/api/library/{job_id}/thumb")
async def set_library_thumb(job_id: str, request: Request):
    """Attach the browser-rendered 3D snapshot (a data: URL) to a model row.
    Only the client can produce this, so it's sent up after the viewer renders."""
    body = await request.body()
    if len(body) > 4 * 1024 * 1024:
        return _error(413, "Thumbnail too large.")
    try:
        thumb = json.loads(body).get("thumb")
    except Exception:
        return _error(400, "Invalid JSON body.")
    if not isinstance(thumb, str) or not thumb.startswith("data:image/"):
        return _error(400, "Missing or invalid thumb (expected a data:image/... URL).")
    if db.set_thumb(job_id, thumb):
        return {"ok": True}
    return _error(404, "Unknown model id.")


# --------------------------------------------------------------------------
# Static frontend (mounted last so it doesn't shadow /api/*).
# --------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
