# SOLIDIFY v2 — Frontend Implementation (multi-phase)

Sci-fi "operator terminal" frontend for the TRELLIS.2 image-to-3D web app,
ported from the Claude Design prototype **Solidify v2.dc.html** and wired to the
existing FastAPI backend. Branch: `web-solidify-v2`.

## Decisions (locked with owner)
- **Fidelity:** pixel-faithful on desktop (Vanta network hero, glitch/decrypt
  text, scan-lines, specimen-scan). On phones / reduced-motion, downgrade to
  lighter FX (no Vanta, no constant glitch loops).
- **Auth:** wire the branded login to the existing HTTP Basic Auth
  (`TRELLIS_AUTH_USER` / `TRELLIS_AUTH_PASSWORD`); no new accounts yet.
- **Phasing:** all four screens themed first, then deepen backend wiring.

## Files
| File | Role |
|---|---|
| `server/static/index.html` | Shell: fonts, three/vanta/gsap, `fx.js`, `model-preview.js`, global theme + keyframes, hover/focus classes, `#app` root |
| `server/static/app.js` | Controller: state machine, render functions for Login/Home/About/Studio, real backend wiring, `?demo=1` simulation |
| `server/static/fx.js` | Effect web components (`glitch-text`, `letter-glitch`, `vanta-bg`, `specimen-scan`) — ported verbatim + mobile downgrade |
| `server/static/model-preview.js` | three.js `<model-preview>` viewer (`forming`/`orbit`); loads the real GLB via `src` |
| `server/static/assets/{biker,frog,frog-mesh}.png` | Design images (login specimen, field-test input/output) |
| `server/main.py` | Basic-Auth middleware now exempts `/` + `/static/*`; `/api/*` stays gated |

## Motion / polish libraries (loaded via CDN in `index.html`)
- **three.js 0.134 + Vanta NET** — animated network hero background.
- **GSAP + ScrollTrigger** — scroll-triggered reveals (`[data-reveal]` elements
  animate in as they enter the viewport; re-armed each render).
- **Lenis** — site-wide smooth scroll, integrated with the GSAP ticker /
  ScrollTrigger. Disabled on mobile / reduced-motion.
- **react-bits-style touches (hand-ported, no React):** `glitch-text` decrypt,
  `letter-glitch` canvas, **SpotlightCard** pointer-follow glow (`[data-spot]`),
  and **Magnet** CTAs (`[data-magnet]`). All gated off on mobile / reduced-motion;
  3D viewers carry `data-lenis-prevent` so scroll-zoom works over the canvas.

## Backend contract (unchanged)
`GET /api/config` · `GET /api/health` · `POST /api/jobs` (image, seed,
target_faces, pipeline_type, skip_printable; 429 if queue full) · poll
`GET /api/jobs/{id}` · download `GET /api/jobs/{id}/files/{name}`.

State machine: `queued → loading_model → preprocessing → generating →
making_printable → done | error`.

Mapping: design stage chips QUEUED/PREPARING/GENERATING/FINALIZING ←
queued / preprocessing / generating / making_printable; `loading_model` → the
"warming" screen; 429 → the "busy" screen; advanced **quality** draft/standard/high
→ `pipeline_type` 512/1024/1024_cascade; **mesh detail** → `target_faces`.

## Phases

### Phase 0 — Branch ✅
`git checkout -b web-solidify-v2`.

### Phase 1 — Theme + shell + all screens ✅
Pixel-faithful Login, Home, About, Studio render with the full theme, fonts, and
effects. Responsive downgrade for phones / `prefers-reduced-motion` (fx.js skips
Vanta + freezes glitch loops). Full-res design images in place.

### Phase 2 — Studio wired to the real backend ✅
`generate()` → `POST /api/jobs` (mapped params); 429 → busy screen; 2s polling of
`/api/jobs/{id}` drives phase + stage chips + progress + live log; `done` builds
real model-info / print-diagnostics rows; `error` shows message + verbatim hint
with retry; downloads hit the real file endpoints; `localStorage` restores an
in-flight job on reload.

### Phase 3 — Auth + real 3D preview ✅
Login `submitLogin` validates via `/api/health` with a Basic `Authorization`
header, stores the credential in memory, and attaches it to every `/api` call;
401 → ACCESS DENIED state; logout clears it. Middleware exempts the static shell
so the branded login is the gate (no native popup). `<model-preview>` orbit mode
loads the job's real GLB with the cyan wireframe styling; `forming` keeps the
placeholder as a loading visual.

### Phase 5 — Full backend wiring + Library (2026-07-23) ✅ (verified vs simulator)
- Auth-aware file access: `apiFileURL()` fetches API files WITH the Basic-Auth
  header → blob URL, used for the orbit GLB preview and STL/GLB downloads (plain
  `<a>`/GLTFLoader GETs would 401 when `TRELLIS_AUTH_PASSWORD` is set). Import map
  added so three.js example loaders resolve `"three"`. Normals computed on load
  so vertex-colored/normal-less GLBs render lit, not black.
- Queue UX: live header count ("N JOBS WAITING", "YOUR JOB RUNNING", "FULL");
  `queued` → auto-advances to generating when your turn comes; 429 → "LAB AT
  CAPACITY" with auto-retry (resubmits when a slot frees) + Cancel; Cancel button
  to leave the queue. Live **PIPELINE NOW** activity panel on Studio idle lists
  every in-flight job (from `GET /api/jobs`).
- **Library** (`server/static/app.js`): client-side gallery in localStorage.
  On completion a 3D snapshot is captured from the WebGL canvas
  (`model-preview.snapshot()`, needs `preserveDrawingBuffer`) and stored with
  stats + auth-aware STL/GLB download links. New LIBRARY nav + screen.
- Verified end-to-end against a faithful backend **simulator** (scratchpad
  `devserve.py`, serving a real generated GLB) driving the REAL non-demo code
  path: login → upload → queued → generating → done → real GLB preview →
  downloads → Library snapshot; plus 429/busy, warming, and error+hint.
  NOTE: real TRELLIS inference is Mac-only; the simulator returns one canned mesh
  for every job (so "frog.png" showed the biker mesh — a test artifact, not a bug).

### Phase 4 — Polish + verification 🔄
Verified in-browser (desktop, `?demo=1`, via Chrome MCP):
- [x] Login (biker specimen, glitch title, form) + sign-in
- [x] Home (Vanta network hero, glitch headline, specimen-scan, stat row, protocol)
- [x] About (letter-glitch canvas, dossier cards)
- [x] Studio idle (dropzone) → upload → uploaded (source photo + generate + advanced)
- [x] Studio running (live %, stage chips, forming 3D preview, live-feed log)
- [x] Studio success (orbit 3D preview, model info, diagnostics, downloads)
- [x] JS syntax check (node --check) on app.js / fx.js / model-preview.js — clean
- [x] Console clean except a harmless "multiple Three.js instances" warning
      (Vanta uses three 0.134 global; model-preview imports 0.161 module)

Remaining (need the real backend / a phone):
- [ ] busy (429) + warming (loading_model) + error(+hint) states — trigger on the
      real backend (demo happy-path doesn't hit them; markup mirrors verified screens)
- [ ] Confirm phone downgrade + reduced-motion on a small viewport
- [ ] End-to-end run on the Mac GPU (real inference) — upload→result→download,
      real GLB into the orbit preview

## Verify locally
```bash
bash run_server.sh           # or: uvicorn server.main:app --port 8000
# open http://127.0.0.1:8000
# off-GPU (Windows): http://127.0.0.1:8000/?demo=1  → preview working/success/error
```
On the dev Windows client, `/api/config` + `/api/health` work; real inference is
Mac-only, so use `?demo=1` to exercise the working/success/error screens there.
With `TRELLIS_AUTH_PASSWORD` set: wrong key → ACCESS DENIED; correct key → Home;
`/api/*` is 401 without the header while `/` still loads.
