#!/usr/bin/env bash
#
# Launch the TRELLIS.2 web server (FastAPI + uvicorn).
#
# SINGLE WORKER ONLY. We deliberately do NOT pass --workers > 1: each uvicorn
# worker would be a separate process with its own in-memory JobStore AND its
# own resident TRELLIS.2 pipeline. Multiple pipelines would contend for the one
# MPS device (blowing up thermal throttling) and defeat the whole serialized
# single-worker-thread design. One process, one pipeline, one job at a time.
#
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p jobs

HOST="${TRELLIS_HOST:-0.0.0.0}"
PORT="${TRELLIS_PORT:-8000}"

echo "============================================================"
echo "TRELLIS.2 web server"
echo "  TRELLIS_MODEL_ID     = ${TRELLIS_MODEL_ID:-microsoft/TRELLIS.2-4B}"
echo "  TRELLIS_DEVICE       = ${TRELLIS_DEVICE:-mps}"
echo "  TRELLIS_PIPELINE_TYPE= ${TRELLIS_PIPELINE_TYPE:-1024_cascade}"
echo "  TRELLIS_NO_TEXTURE   = ${TRELLIS_NO_TEXTURE:-1}"
echo "  Listening on         = http://${HOST}:${PORT}"
echo "  Database             = ${TRELLIS_DB_PATH:-$(pwd)/trellis.db}"
echo "  Auth                 = managed by the operators table in the DB."
echo "                          0 operators -> OPEN (bootstrap); add one with"
echo "                          'python -m server.admin add <name>' to require login."
echo "============================================================"

exec uvicorn server.main:app --host "${HOST}" --port "${PORT}"
