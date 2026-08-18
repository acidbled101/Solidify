#!/usr/bin/env bash
#
# Expose the local TRELLIS.2 web server to the internet via a Cloudflare
# "quick tunnel" -- no account, no router/port-forwarding config, no domain
# needed. Prints a random https://*.trycloudflare.com URL that proxies to
# your local server for as long as this script keeps running.
#
# IMPORTANT: run_server.sh must already be running in another terminal, and
# TRELLIS_AUTH_PASSWORD should be set on it before you run this -- anyone
# with the tunnel URL can reach the server, and job submission has no other
# rate limiting.
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${TRELLIS_PORT:-8000}"

if ! command -v cloudflared &>/dev/null; then
    echo "cloudflared not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Install it from https://brew.sh, or install" >&2
        echo "cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
        exit 1
    fi
    brew install cloudflared
fi

if ! curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    echo "Warning: nothing is answering on http://127.0.0.1:${PORT} yet." >&2
    echo "Start the server first (bash run_server.sh) in another terminal." >&2
fi

echo "============================================================"
echo "Starting a Cloudflare quick tunnel to http://127.0.0.1:${PORT}"
echo "Your public URL will appear below (https://*.trycloudflare.com)."
echo "Press Ctrl+C to stop sharing -- the URL stops working immediately."
echo "============================================================"

exec cloudflared tunnel --url "http://127.0.0.1:${PORT}"
