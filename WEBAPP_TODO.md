# TRELLIS.2 Web App — Full Plan & Deployment Handover

> **Purpose:** the complete plan for the TRELLIS.2 web app on the
> `web-app-fastapi` branch. Written on the **Windows** side (the client) so the
> **Mac** side (the inference server) can execute it end to end.
>
> **If you are Claude Code running on the Mac:** do **Section 1 (set up & run the
> app locally)** then **Section 2 (expose it with Tailscale Funnel)**, checking
> the boxes as you go. Report the public URL back to the owner. Sections 3–5 are
> web-app changes that happen on the Windows side once the owner provides a design
> theme — leave them unless asked.

---

## Context / the goal (read first)

> 🧠 **CORE PRINCIPLE — do not lose sight of this:** **ALL inference runs on the
> Mac's GPU.** The website is **only an interface / remote control** for the model
> running on this Mac. The frontend never computes anything — it uploads an image,
> the **Mac** generates the 3D model, and the frontend shows/downloads the result.
> Nothing (not Vercel, not the client device) does the generation except this Mac.

- **What this is:** a FastAPI web app (`server/`) that turns an uploaded photo
  into a print-ready 3D model (`.stl` / `.glb`). Frontend is static
  (`server/static/index.html` + `app.js`). The frontend is **just the interface**;
  the Mac does the actual work.
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
| Custom domain | **Not now** | Won't pay ~$10/yr until the tool proves it gets used. Pretty domain is an optional later upgrade, no code changes |
| Vercel in front | **Not now** (optional cosmetic later) | Vercel can't run the model and doesn't remove the tunnel; only benefit is a prettier URL. See Section 5 |
| Auth | **Per-user login page** (whitelist of accounts) | Owner wants to add/remove individual professors; branded login, not the browser's basic-auth popup |
| Inference | **Stays on the Mac GPU**, unchanged | Only the Mac can run the model |
| Design | Full theme **pending** from owner | Login page will be built to match, so it's styled once |

---

## Section 1 — Set up & run the app on the Mac (MAC — do this first)

Goal: get the web app running locally on the Mac and confirm it generates a model,
**before** exposing it to the internet.

**Prerequisites (one-time):** macOS on Apple Silicon (M1+), Python 3.11+, ~15 GB
free disk for weights, and a HuggingFace account with access to the gated models
(approval is usually instant):
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `briaai/RMBG-2.0`
- TRELLIS.2 weights

- [ ] **Get the code and switch to this branch:**
  ```bash
  git clone https://github.com/acidbled101/Solidify.git
  cd Solidify
  git checkout web-app-fastapi
  # (If the repo already exists on this Mac: cd into it, then
  #  git checkout web-app-fastapi && git pull)
  ```
- [ ] **(Recommended) Download the Metal toolchain** for the fast texture baker
      (optional; setup falls back to a slower pure-Python baker without it):
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
- [ ] **Log into HuggingFace** and request access to the 3 gated models above:
  ```bash
  hf auth login
  ```
- [ ] **Run setup** — creates `.venv`, installs deps, clones & patches TRELLIS.2,
      builds the Metal backends. Takes a while:
  ```bash
  bash setup.sh
  # To skip the Metal build (older hardware / faster setup): SKIP_METAL=1 bash setup.sh
  ```
- [ ] **Activate the environment:**
  ```bash
  source .venv/bin/activate
  ```
- [ ] **Start the server** and confirm it comes up:
  ```bash
  bash run_server.sh        # serves on 0.0.0.0:8000
  ```
  In another terminal: `curl -s http://127.0.0.1:8000/api/health` should return
  `{"status":"ok",...}`.
- [ ] **See it in a browser on the Mac:** open **http://localhost:8000**, upload a
      photo, hit **Generate**, and confirm you get a downloadable print-ready model.

> First-ever run downloads ~15 GB of weights (one time, network-bound) and the
> pipeline warm-up takes ~100 s before the first job. After that it stays resident
> and fast. If a run is unusually slow, let the Mac cool down — it throttles hard
> under sustained load (see README "thermal throttling").

---

## Section 2 — Expose it with Tailscale Funnel (MAC — do this after Section 1 works)

Goal: a **permanent, free, public HTTPS URL** (e.g.
`https://<machine>.<tailnet>.ts.net`) that reaches this Mac's server, so
professors can open it from anywhere. Only the Mac installs Tailscale; clients don't.

> ⚠️ **Security order matters:** turn on auth on the server **before** exposing it
> publicly. Until the per-user login page (Section 3) is built, use the built-in
> HTTP Basic Auth as a stopgap by setting `TRELLIS_AUTH_PASSWORD` (and optionally
> `TRELLIS_AUTH_USER`) before running `run_server.sh`. **Never run Funnel with
> auth off.**

- [ ] **Install Tailscale on the Mac** and sign in:
  - `brew install --cask tailscale` (GUI app) **or** `brew install tailscale`
    (CLI + `tailscaled`), then `tailscale up`. Use whichever gives a working
    `tailscale` CLI — Funnel is driven from the CLI.
- [ ] **Enable Funnel prerequisites in the Tailscale admin console:**
  - MagicDNS + HTTPS certificates enabled for the tailnet.
  - **Funnel enabled** for this node (Access Controls / node attributes — the
    console prompts / links you the first time you run a funnel command).
- [ ] **Restart the server WITH auth** (stopgap Basic Auth until Section 3 ships):
  ```bash
  cd <repo root>
  export TRELLIS_AUTH_USER=admin
  export TRELLIS_AUTH_PASSWORD='pick-a-strong-password'
  bash run_server.sh
  ```
- [ ] **Expose port 8000 via Funnel** (second terminal):
  ```bash
  tailscale funnel --bg 8000   # background; prints the public https URL
  tailscale funnel status      # shows the public URL + mapping
  ```
  (Funnel serves publicly on 443 and proxies to local 8000 — automatic.)
- [ ] **Test from an external device** (phone on cellular, NOT home WiFi): open
      the `https://<machine>.<tailnet>.ts.net` URL, confirm the login prompt
      appears, log in, and the page loads and can generate.
- [ ] **Record the public URL** and report it to the owner:
  - Public URL: `__________________________________`
- [ ] **Make it survive reboots:** set up a `launchd` / login-item so `tailscaled`,
      `run_server.sh`, and `tailscale funnel --bg 8000` all restart on boot — the
      whole point is a link that keeps working without babysitting.

### Mac-side gotchas
- Funnel public ports are limited to 443/8443/10000; `tailscale funnel 8000`
  handles the mapping to 443, so the professor-facing URL is clean https.
- Keep `run_server.sh` **single worker** (it already is) — never add `--workers`;
  multiple pipelines would fight over the one GPU.
- Server has a queue cap (`TRELLIS_MAX_QUEUE_DEPTH`, default 10) and upload cap
  (`TRELLIS_MAX_UPLOAD_MB`, default 20) — fine for a small trusted group; tune in
  `server/config.py`.

---

## Section 3 — Web app: per-user login + whitelist (WINDOWS side — pending design)

Build a proper login so access is per-professor, not one shared password. Blocked
on the owner's design theme (so the login page is styled to match — build once).
This replaces the Section 2 stopgap Basic Auth.

- [ ] Branded login page (matches the new design theme, TBD).
- [ ] User whitelist the owner manages (add/remove a professor by editing a list).
      Decide storage: a simple `users` file/table + hashed passwords.
- [ ] Session handling (login → cookie/session) so professors don't re-auth every
      request.
- [ ] Logout + "wrong password" UX.
- [ ] Initial whitelist = the professor list the owner will provide.

## Section 4 — Web app: redesign + features (WINDOWS side — pending design)

Blocked on the owner's full design theme.

- [ ] Visual redesign of the main page (colors, layout, fonts, branding).
- [ ] Text/branding pass — current copy says "Runs locally on this Mac's GPU";
      reword for the professor-facing audience; rename/brand as the owner wants.
- [ ] New features/controls (TBD — e.g. drag-and-drop upload, job history/gallery,
      clearer progress, print-oriented options).
- [ ] **Open question — what does "print" mean?** Today the app produces a
      downloadable print-ready `.stl` (the professor prints it in their own
      slicer). If the owner wants the site to drive a printer directly
      (OctoPrint / Klipper / Bambu), that's a separate feature to scope later.

## Section 5 — OPTIONAL LATER: prettier URL via Vercel proxy (non-blocking)

Only if the tool proves popular and the owner wants `yourapp.vercel.app` instead
of the `*.ts.net` Funnel URL. **Not needed to ship.** Notes / caveats:
- Vercel has **no GPU** and its functions time out in seconds — it **cannot** run
  the model and **cannot** replace the Mac. It can only sit *in front* as a proxy.
- It does **not** remove the need for the Tailscale Funnel — Vercel would proxy
  *to* the Funnel URL. So it's a pure cosmetic layer for the URL.
- Watch: large STL/GLB downloads flowing **through** Vercel hit its bandwidth /
  function limits. Fine at small scale, a thing to monitor.
- Do **not** split the frontend onto Vercel and the API on the Mac as two separate
  URLs — that adds CORS + two-URL sync for zero functional gain.

---

## Status log
- Web app confirmed running locally on the Windows client (`http://127.0.0.1:8000`)
  for design iteration — frontend + `/api/config` + `/api/health` work; only
  actual model inference is unavailable off-Mac (expected).
- Awaiting from owner: **design theme** + **list of professors** for the whitelist.
