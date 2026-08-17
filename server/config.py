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
DEFAULT_TARGET_FACES = int(os.environ.get("TRELLIS_TARGET_FACES", "1000000"))
DEFAULT_SEED = int(os.environ.get("TRELLIS_SEED", "42"))
DEFAULT_TEXTURE_SIZE = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "2048"))

# The server ALWAYS runs generation with no_texture=True by default: it is
# faster, skips the most failure-prone stage, and make_printable's default
# solid-infill voxel remesh discards textures anyway. Overridable, but the
# product decision is to keep it on.
NO_TEXTURE = _as_bool(os.environ.get("TRELLIS_NO_TEXTURE", "1"))

# Skip texture-SLat *inference* (not just the export): a full ~1.3B-param
# diffusion phase + texture decode that are pure waste whenever NO_TEXTURE is on.
# Now ON by default -- A/B verified the output mesh is byte-identical to a full
# run while cutting generation ~35% at 1024 (1032s -> 667s on a real photo) and
# lowering peak memory. Set TRELLIS_SKIP_TEXTURE=0 to force the old full path.
SKIP_TEXTURE_BY_DEFAULT = _as_bool(os.environ.get("TRELLIS_SKIP_TEXTURE", "1"))

# --- Which shape-SLat weights generate ---------------------------------------
# "tuned" runs the flow model with a LoRA adapter fine-tuned on 300 curated
# Thingi10K meshes; "base" disables the adapter, which makes the model
# bit-for-bit the stock TRELLIS.2 weights again -- the adapter is additive with
# a zero-initialised B matrix, so switching it off is exact, not approximate.
# Both variants share ONE loaded pipeline; there is no second copy of a 1.3B
# model and switching costs nothing.
#
# Measured on raw output over a 5-image paired comparison (same seeds, same
# sparse structure, no repair applied): non-manifold edges 0.923% -> 0.515%
# (-44%), separate components 6301 -> 3800 (-40%), detail unchanged.
#
# CAVEAT: that comparison sampled the 512 flow model and decoded at 512. The
# server's default pipeline_type is 1024_cascade, whose second stage
# (shape_slat_flow_model_1024) is NOT adapted, so the measured gain may be
# diluted under the cascade. Only the 512 model is wrapped -- the 1024
# checkpoint has identical layer shapes and would accept these weights without
# raising anything, which is exactly why it must not be given them.
MODEL_VARIANT = os.environ.get("TRELLIS_MODEL_VARIANT", "tuned")
ALLOWED_MODEL_VARIANTS = ("tuned", "base")
ADAPTER_PATH = Path(os.environ.get(
    "TRELLIS_ADAPTER_PATH", str(REPO_ROOT / "adapters" / "sft-1200-1050.pt")))
ADAPTER_RANK = int(os.environ.get("TRELLIS_ADAPTER_RANK", "16"))
ADAPTER_ALPHA = float(os.environ.get("TRELLIS_ADAPTER_ALPHA", "32"))

# --- make_printable defaults -----------------------------------------------
PRINTABLE_TARGET_FACES = int(os.environ.get("TRELLIS_PRINTABLE_TARGET_FACES", "1000000"))
PRINTABLE_OVERHANG_ANGLE = float(os.environ.get("TRELLIS_OVERHANG_ANGLE", "45.0"))
PRINTABLE_SOLID_INFILL = _as_bool(os.environ.get("TRELLIS_SOLID_INFILL", "1"))
SKIP_PRINTABLE_BY_DEFAULT = _as_bool(os.environ.get("TRELLIS_SKIP_PRINTABLE", "0"))

# Which print-prep implementation runs. Measured on real job meshes, v3 beats v1
# on every axis -- ~2x faster, 4-7x smaller files, 20x closer to the original
# surface, and watertight where v1 was not (see POSTPROC_V2.md) -- so it is the
# default. v1 stays selectable per job (and via TRELLIS_PRINTABLE_PIPELINE) as
# the fallback: it needs only trimesh, while v2/v3 need the optional
# pymeshlab/manifold3d/meshlib extras.
#   v1 -- trellis_core/printable.py: trimesh repair + voxel flood-fill
#   v2 -- printable_v2.py: PyMeshLab clean/Taubin + Manifold3D
#   v3 -- printable_v2.py with MeshLib orientation repair + SDF rebuild
PRINTABLE_PIPELINE = os.environ.get("TRELLIS_PRINTABLE_PIPELINE", "v3")
ALLOWED_PRINTABLE_PIPELINES = ("v1", "v2", "v3")

# --- Server / networking ----------------------------------------------------
HOST = os.environ.get("TRELLIS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRELLIS_PORT", "8000"))
JOBS_DIR = Path(os.environ.get("TRELLIS_JOBS_DIR", str(REPO_ROOT / "jobs")))
MAX_UPLOAD_MB = int(os.environ.get("TRELLIS_MAX_UPLOAD_MB", "20"))

# --- Local database ---------------------------------------------------------
# One SQLite file on this Mac: the operator allow-list + a persistent archive of
# every completed generation. See server/db.py.
DB_PATH = Path(os.environ.get("TRELLIS_DB_PATH", str(REPO_ROOT / "trellis.db")))

# Shown as the per-job time estimate ("~N MIN") until enough real jobs exist for
# the live DB average to take over.
DEFAULT_JOB_MINUTES = float(os.environ.get("TRELLIS_DEFAULT_JOB_MINUTES", "6"))

# --- Validation allowlists --------------------------------------------------
ALLOWED_PIPELINE_TYPES = ("512", "1024", "1024_cascade")
ALLOWED_TEXTURE_SIZES = (512, 1024, 2048)

# When True, API error responses include full tracebacks. Off by default so we
# don't dump internals to LAN users.
DEBUG = _as_bool(os.environ.get("TRELLIS_DEBUG", "0"))

# --- Auth -------------------------------------------------------------------
# Authentication is now driven by the operators table in the local DB (managed
# with `python -m server.admin`), NOT by these env vars:
#   * 0 operators  -> open "bootstrap" mode (so you can't lock yourself out).
#   * >= 1 operator -> every /api/* route requires a real login.
# See server/main.py (auth middleware) and server/db.py.
#
# TRELLIS_AUTH_USER / TRELLIS_AUTH_PASSWORD are kept ONLY as an optional
# convenience: if both are set and the operators table is still empty at
# startup, that one account is seeded automatically (handy for launchd). After
# that, manage operators with the admin CLI.
AUTH_SEED_USER = os.environ.get("TRELLIS_AUTH_USER", "")
AUTH_SEED_PASSWORD = os.environ.get("TRELLIS_AUTH_PASSWORD", "")

# How long a login token stays valid before the operator must sign in again.
SESSION_TTL_DAYS = int(os.environ.get("TRELLIS_SESSION_TTL_DAYS", "30"))

# Hard cap on queued (not yet started) jobs. Once reached, new submissions are
# rejected with 429 instead of piling up indefinitely -- relevant once the
# server is reachable by more than just people you trust.
MAX_QUEUE_DEPTH = int(os.environ.get("TRELLIS_MAX_QUEUE_DEPTH", "10"))
