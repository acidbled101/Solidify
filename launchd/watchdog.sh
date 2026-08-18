#!/bin/bash
#
# Health watchdog for the TRELLIS web server.
#
# WHAT THIS CATCHES THAT KeepAlive DOES NOT
# -----------------------------------------
# launchd's KeepAlive restarts a process that EXITS. It cannot see a process
# that is still alive but wedged -- which is the failure this machine actually
# produces: the macOS GPU watchdog kills a long-running Metal kernel, the
# decoder is left holding a dead command buffer, and uvicorn keeps its socket
# open while answering nothing. To launchd that server is perfectly healthy.
#
# So this polls /api/health and kickstarts the job when it stops answering.
#
# WHY THREE STRIKES, NOT ONE
# --------------------------
# A generation job occupies the single worker thread for minutes at a time and
# the machine is under heavy GPU load throughout. A single slow response is
# normal operation, not a fault. Restarting on it would kill a job a user is
# waiting on -- the watchdog would become the outage. Three consecutive misses
# roughly three minutes apart is a genuinely stuck server.
#
# The strike counter lives in a file because launchd runs this as a fresh
# process every interval; there is no memory between runs.

set -u

URL="http://127.0.0.1:8000/api/health"
STATE="${TMPDIR:-/tmp}/trellis_watchdog_strikes"
LABEL="gui/$(id -u)/com.trellis.webserver"
MAX_STRIKES=3
# Generous: the server may be mid-generation and slow to answer, and a timeout
# here counts as a strike. Long enough not to cry wolf, short enough that three
# strikes still resolves inside ~5 minutes.
TIMEOUT=20

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') watchdog: $*"; }

# 401 means auth is on and the server is answering -- that is healthy. Any 2xx
# or 4xx proves the process is alive and serving; only a connection failure,
# timeout or 5xx counts against it.
#
# No `|| echo 000` fallback here: curl's -w already prints 000 when it cannot
# connect, so adding one concatenates two codes. That produced "000000" on a
# dead port, and would have produced "200000" for a healthy server whose curl
# exited non-zero for any other reason -- which fails the numeric test below and
# would restart a server that was working perfectly.
code=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" "$URL" 2>/dev/null)

if [[ "$code" =~ ^[0-9]{3}$ ]] && [ "$code" -ge 100 ] && [ "$code" -lt 500 ]; then
  if [ -s "$STATE" ]; then
    log "healthy again (HTTP $code), clearing $(cat "$STATE") strike(s)"
  fi
  : > "$STATE"
  exit 0
fi

strikes=$(cat "$STATE" 2>/dev/null || echo 0)
strikes=$((strikes + 1))
echo "$strikes" > "$STATE"
log "no healthy response (HTTP $code) -- strike $strikes/$MAX_STRIKES"

if [ "$strikes" -ge "$MAX_STRIKES" ]; then
  log "restarting $LABEL"
  launchctl kickstart -k "$LABEL"
  : > "$STATE"
fi
