# TRELLIS.2 Web App — TODO & Deployment Handover

> **Purpose of this file:** A running checklist for the TRELLIS.2 web app on the
> `web-app-fastapi` branch. Written on the **Windows** side (the client) so the
> **Mac** side (the inference server) can pick up the deployment tasks.
>
> **If you are Claude Code running on the Mac:** your job right now is
> **Section 1 — Deployment (Tailscale Funnel)**. Do those steps, check the boxes,
> and report the public URL back. Sections 2–3 are web-app changes that happen on
> the Windows side once the owner provides a design theme — you don't need to
> touch those unless asked.

---

## Context / the goal (read first)

- **What this is:** a FastAPI web app (`server/`) that turns an uploaded photo
  into a print-ready 3D model (`.stl` / `.glb`). The frontend is static
  (`server/static/index.html` + `app.js`).
- **Who runs what:**
  - **Mac** = the server. Holds the model, does all generation on the MPS GPU,
    runs `bash run_server.sh` (binds `0.0.0.0:8000`, single worker, one job at a
    time — see the header comment in `run_server.sh`).
  - **Any other device** (owner's Windows PC, professors' laptops/phones) = just
    a browser client hitting the server's URL. Installs nothing.
- **End goal:** hand professors a **link + login** so they can generate 3D files
  from anywhere with **zero install and zero config**. If they see setup work,
  they won't use it.

### Decisions already made (do not re-litigate)

| Topic | Decision | Why |
|---|---|---|
| Remote access | **Public tunnel**, not a per-device VPN | Professors must install nothing — just click a link |
| Stable URL | **Tailscale Funnel** (free) | Free + permanent URL; only the **Mac** installs anything |
| Custom domain | **Not now** | Won't pay ~$10/yr until the tool proves it gets used. Pretty domain (`trellis.example.com`) is an optional later upgrade, no code changes |
| Auth | **Per-user login page** (whitelist of accounts) | Owner wants to add/remove individual professors; branded login, not the browser's basic-auth popup |
| Inference | **Stays on the Mac GPU**, unchanged | Only the Mac can run the model |
| Design | Full theme **pending** from owner | Login page will be built to match, so it's styled once |

---

## Section 1 — Deployment: Tailscale Funnel (MAC — do this now)

Goal: a **permanent, free, public HTTPS URL** (e.g.
`https://<machine>.<tailnet>.ts.net`) that reaches this Mac's server, so
professors can open it from anywhere. Only the Mac needs Tailscale; clients don't.

> ⚠️ **Security order matters:** turn on auth on the server **before** exposing it
> to the public internet with Funnel. Until the per-user login page (Section 2) is
> built, use the built-in HTTP Basic Auth as a stopgap by setting
> `TRELLIS_AUTH_PASSWORD` (and optionally `TRELLIS_AUTH_USER`) before running
> `run_server.sh`. Never run Funnel with auth off.

- [ ] **Install Tailscale on the Mac** and sign in.
  - `brew install --cask tailscale` (GUI app) **or** `brew install tailscale`
    (CLI + `tailscaled`), then `tailscale up`. Use whichever gives a working
    `tailscale` CLI — Funnel is driven from the CLI.
- [ ] **Enable the prerequisites in the Tailscale admin console** (needed for
      Funnel's public HTTPS):
  - MagicDNS + HTTPS certificates enabled for the tailnet.
  - **Funnel enabled** for this node (Access Controls / node attributes — the
    admin console will prompt / link you the first time you run a funnel command).
- [ ] **Start the server WITH auth** (stopgap Basic Auth until Section 2 ships):
  ```bash
  cd <repo root>
  export TRELLIS_AUTH_USER=admin
  export TRELLIS_AUTH_PASSWORD='pick-a-strong-password'
  bash run_server.sh          # serves on 0.0.0.0:8000
  ```
  Confirm `curl -s http://127.0.0.1:8000/api/health` returns `{"status":"ok",...}`.
- [ ] **Expose port 8000 via Funnel** (in a second terminal):
  ```bash
  tailscale funnel --bg 8000   # background; prints the public https URL
  tailscale funnel status      # shows the public URL + mapping
  ```
  (Funnel serves publicly on 443 and proxies to local 8000 — that's automatic.)
- [ ] **Test from an external device** (e.g. phone on cellular, NOT home WiFi):
      open the `https://<machine>.<tailnet>.ts.net` URL, confirm the login prompt
      appears, log in, and the page loads.
- [ ] **Record the public URL** below and report it back to the owner:
  - Public URL: `__________________________________`
- [ ] **Note the persistence behavior:** if the Mac reboots, confirm `tailscaled`
      and the funnel come back automatically (re-run `tailscale funnel --bg 8000`
      if not). Consider a login-item / launchd entry so `run_server.sh` + funnel
      restart on boot — the whole point is a link that keeps working.

### Gotchas / notes for the Mac side
- Funnel public ports are limited to 443/8443/10000; `tailscale funnel 8000`
  handles the mapping to 443 for you, so the professor-facing URL is clean https.
- Keep `run_server.sh` as **single worker** (it already is) — do not add
  `--workers`; multiple pipelines would fight over the one GPU.
- The server has a queue cap (`TRELLIS_MAX_QUEUE_DEPTH`, default 10) and an
  upload size cap (`TRELLIS_MAX_UPLOAD_MB`, default 20) — fine defaults for a
  small trusted group; see `server/config.py` to tune.

---

## Section 2 — Web app: per-user login + whitelist (WINDOWS side — pending design)

Build a proper login so access is per-professor, not one shared password. Blocked
on the owner's design theme (so the login page is styled to match — build once).

- [ ] Branded login page (matches the new design theme, TBD).
- [ ] User whitelist the owner manages (add/remove a professor by editing a list).
      Decide storage: a simple `users` file/table + hashed passwords.
- [ ] Session handling (login → cookie/session) so professors don't re-auth every
      request; replaces the stopgap HTTP Basic Auth from Section 1.
- [ ] Logout + "wrong password" UX.
- [ ] Initial whitelist = the professor list the owner will provide.

## Section 3 — Web app: redesign + features (WINDOWS side — pending design)

Blocked on the owner's full design theme.

- [ ] Visual redesign of the main page (colors, layout, fonts, branding).
- [ ] Text/branding pass — the current copy says "Runs locally on this Mac's GPU";
      reword for the professor-facing audience; rename/brand as the owner wants.
- [ ] New features/controls (TBD from owner — e.g. drag-and-drop upload, job
      history/gallery, clearer progress, print-oriented options).
- [ ] **Open question — what does "print" mean?** Today the app produces a
      downloadable print-ready `.stl` (the professor prints it in their own
      slicer). If the owner wants the site to drive a printer directly
      (OctoPrint / Klipper / Bambu), that's a separate feature to scope later.

---

## Status log
- Web app confirmed running locally on the Windows client (`http://127.0.0.1:8000`)
  for design iteration — frontend + `/api/config` + `/api/health` work; only
  actual model inference is unavailable off-Mac (expected).
- Awaiting from owner: **design theme** + **list of professors** for the whitelist.
