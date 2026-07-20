"""
Single source of truth for all server defaults.

Every other server module imports values from here instead of hardcoding, so
"update the model / change a default" is a one-line change (usually just an
environment variable set before `bash run_server.sh`).
"""

import os
from pathlib import Path


def _as_bool(value: str) -> bool:
    return value not in ("0", "false", "False", "no", "No", "")


# Repo root = parent of this `server/` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Model / generation defaults -------------------------------------------
MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
DEVICE = os.environ.get("TRELLIS_DEVICE", "mps")
DEFAULT_PIPELINE_TYPE = os.environ.get("TRELLIS_PIPELINE_TYPE", "1024_cascade")
DEFAULT_TARGET_FACES = int(os.environ.get("TRELLIS_TARGET_FACES", "500000"))
DEFAULT_SEED = int(os.environ.get("TRELLIS_SEED", "42"))
DEFAULT_TEXTURE_SIZE = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "2048"))

# The server ALWAYS runs generation with no_texture=True by default: it is
# faster, skips the most failure-prone stage, and make_printable's default
# solid-infill voxel remesh discards textures anyway. Overridable, but the
# product decision is to keep it on.
NO_TEXTURE = _as_bool(os.environ.get("TRELLIS_NO_TEXTURE", "1"))

# --- make_printable defaults -----------------------------------------------
PRINTABLE_TARGET_FACES = int(os.environ.get("TRELLIS_PRINTABLE_TARGET_FACES", "1000000"))
PRINTABLE_OVERHANG_ANGLE = float(os.environ.get("TRELLIS_OVERHANG_ANGLE", "45.0"))
PRINTABLE_SOLID_INFILL = _as_bool(os.environ.get("TRELLIS_SOLID_INFILL", "1"))
SKIP_PRINTABLE_BY_DEFAULT = _as_bool(os.environ.get("TRELLIS_SKIP_PRINTABLE", "0"))

# --- Server / networking ----------------------------------------------------
HOST = os.environ.get("TRELLIS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRELLIS_PORT", "8000"))
JOBS_DIR = Path(os.environ.get("TRELLIS_JOBS_DIR", str(REPO_ROOT / "jobs")))
MAX_UPLOAD_MB = int(os.environ.get("TRELLIS_MAX_UPLOAD_MB", "20"))

# --- Validation allowlists --------------------------------------------------
ALLOWED_PIPELINE_TYPES = ("512", "1024", "1024_cascade")
ALLOWED_TEXTURE_SIZES = (512, 1024, 2048)

# When True, API error responses include full tracebacks. Off by default so we
# don't dump internals to LAN users.
DEBUG = _as_bool(os.environ.get("TRELLIS_DEBUG", "0"))
