"use strict";
/* SOLIDIFY — sci-fi frontend for the TRELLIS.2 image-to-3D web app.
   Ported from the "Solidify v2" Claude Design prototype and wired to the real
   FastAPI backend (/api/config, /api/health, POST /api/jobs, poll, downloads).

   Add ?demo=1 to the URL to drive the Studio with a local simulation (no GPU),
   useful for previewing the working / success / error states off the lab machine. */
(function () {
  var DEMO = new URLSearchParams(location.search).has("demo");
  var LS_KEY = "trellis_job_id";
  var appEl = document.getElementById("app");

  var ASSETS = {
    biker: "/static/assets/biker.png",
    biker3d: "/static/assets/biker-3d.png",
    frog: "/static/assets/frog.png",
    mesh: "/static/assets/frog-mesh.png",
    stl: "/static/assets/specimen-042.stl", // decimated hero teaser (~3MB); real jobs return full-quality output
  };

  // ---- state (mirrors the prototype's DCLogic state) -----------------------
  var state = {
    screen: "login",              // login | home | about | studio
    authName: "", authKey: "", authError: false,
    phase: "idle",                // idle | uploaded | busy | warming | running | success | error
    photoSrc: null, photoName: "", photoSize: "", photoFile: null,
    seed: "", detail: 80000, quality: "standard", skipPrep: false, advOpen: false,
    prog: 0, stageIdx: -1, log: [], queuePos: 0, busyReason: "", errMsg: "", errHint: "", dragOver: false,
    jobId: "",
    infoRows: null, diagRows: null, previewSrc: null, previewPath: null, libDraft: null,
    dlSTL: null, dlSTLName: "", dlGLB: null, dlGLBName: "",
    queueDepth: 0, maxUploadMB: 15,
    funIdx: 0,
    _poll: null, _sim: null, _health: null, _retry: null, _fun: null,
  };

  var serverConfig = null;
  var AUTH = null; // "Basic <base64>" once signed in; attached to every /api call

  // Single source of truth for backend job statuses: chip index, chip label,
  // activity/queue label, whether it's an in-flight stage, and the live-log line.
  // Everything status-derived (chips, stage index, activity rows, logs, the
  // in-flight filter) reads from here so a status can't drift across maps.
  var STATUS = {
    queued:           { idx: 0, chip: "QUEUED",     activity: "QUEUED",     active: true,  log: "> job queued · awaiting the lab machine" },
    loading_model:    { idx: 1, chip: "PREPARING",  activity: "WARMING",    active: true,  log: "> loading AI model into GPU memory…" },
    preprocessing:    { idx: 1, chip: "PREPARING",  activity: "PREPARING",  active: true,  log: "> preparing: silhouette + depth extraction" },
    generating:       { idx: 2, chip: "GENERATING", activity: "GENERATING", active: true,  log: "> generating: volumetric field · marching cubes" },
    making_printable: { idx: 3, chip: "FINALIZING", activity: "FINALIZING", active: true,  log: "> print-prep: sealing holes, thickening walls" },
    done:             { activity: "DONE" },
    error:            { activity: "FAILED" },
  };
  var STAGE_NAMES = ["QUEUED", "PREPARING", "GENERATING", "FINALIZING"];

  // Playful rotating status words shown while a job runs (Claude-Code style).
  var FUN_WORDS = [
    "Cooking", "Beaming", "Materializing", "Sculpting", "Conjuring", "Reticulating splines",
    "Tessellating", "Voxelizing", "Crystallizing", "Fabricating", "Summoning geometry",
    "Baking normals", "Marching cubes", "Extruding", "Densifying", "Rezzing", "Wrangling photons",
    "Unfolding", "Printing matter", "Solidifying",
  ];

  // ---- tiny helpers --------------------------------------------------------
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(n) { return (n == null || isNaN(n)) ? "–" : Number(n).toLocaleString(); }
  function shortId(id) { return String(id || "").slice(0, 4).toUpperCase(); }
  function baseName(name) { return String(name || "").replace(/\.[^.]+$/, ""); }
  function stageLog(st) { return (STATUS[st] && STATUS[st].log) || ("> " + st); }
  function authHeaders(extra) { var h = Object.assign({}, extra || {}); if (AUTH) h.Authorization = AUTH; return h; }
  function api(path, opts) { opts = opts || {}; opts.headers = authHeaders(opts.headers); return fetch(path, opts); }

  // Auth-aware file access. Downloads (<a download>) and the 3D preview's
  // GLTFLoader fetch are plain browser GETs that do NOT carry our in-memory
  // Authorization header, so they would 401 whenever TRELLIS_AUTH_PASSWORD is
  // set. Fetch the file *with* the header and hand back a same-origin blob: URL
  // that any element can use freely. Tracked so we can revoke on reset/logout.
  var _objectUrls = [];
  async function apiFileURL(path) {
    var r = await api(path);
    if (!r.ok) throw new Error("file fetch failed: " + r.status);
    var url = URL.createObjectURL(await r.blob());
    _objectUrls.push(url);
    return url;
  }
  function revokeObjectUrls() {
    _objectUrls.forEach(function (u) { try { URL.revokeObjectURL(u); } catch (e) {} });
    _objectUrls = [];
  }

  // ---- Library (client-side gallery of past generations) -------------------
  // Persisted per-browser in localStorage: each entry holds a 3D snapshot
  // (data URL), stats, and the download paths. Downloads hit the live API, so
  // they work while the lab machine still holds the job (jobs are in-memory).
  var LIB_KEY = "solidify_library";
  function loadLibrary() { try { return JSON.parse(localStorage.getItem(LIB_KEY) || "[]"); } catch (e) { return []; } }
  function saveLibrary(list) { try { localStorage.setItem(LIB_KEY, JSON.stringify(list.slice(0, 40))); } catch (e) { /* quota — oldest dropped by slice */ } }
  function addToLibrary(entry) {
    var list = loadLibrary().filter(function (e) { return e.id !== entry.id; });
    list.unshift(entry);
    saveLibrary(list);
    if (state.screen === "library") render();
  }
  function removeFromLibrary(id) { saveLibrary(loadLibrary().filter(function (e) { return e.id !== id; })); }
  function fmtDate(iso) {
    try { var d = new Date(iso); return d.toLocaleDateString() + " · " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
    catch (e) { return ""; }
  }

  // =========================================================================
  // Config / health
  // =========================================================================
  async function loadConfig() {
    try {
      var r = await api("/api/config");
      if (!r.ok) return false; // 401 pre-login when auth is on — caller keeps showing login
      serverConfig = await r.json();
      if (serverConfig) {
        var df = parseInt(serverConfig.default_target_faces, 10);
        if (!isNaN(df)) state.detail = Math.max(20000, Math.min(200000, df));
        if (serverConfig.skip_printable_by_default) state.skipPrep = true;
        if (serverConfig.max_upload_mb) state.maxUploadMB = serverConfig.max_upload_mb;
      }
      refreshHealth();
      return true;
    } catch (e) { return false; /* keep sensible defaults */ }
  }
  async function refreshHealth() {
    try {
      var h = await (await api("/api/health")).json();
      state.queueDepth = h.queue_depth || 0;
      patchQueueHeader();
    } catch (e) { /* health failing shouldn't block the UI */ }
  }
  function patchQueueHeader() {
    var el = document.getElementById("hdr-queue");
    if (el) el.textContent = queueLabelFor();
  }
  // Keep the header queue count live while signed in (every 6s), without
  // re-rendering the page (which would disturb scroll / the 3D canvas). On the
  // studio-idle screen the activity fetch already returns the queue, so we use
  // it as the single source and skip the parallel /api/health call there.
  function startHealthPoll() {
    stopHealthPoll();
    state._health = setInterval(function () {
      if (state.screen === "login") return;
      if (state.screen === "studio" && state.phase === "idle") refreshActivity();
      else refreshHealth();
    }, 6000);
  }
  function stopHealthPoll() { if (state._health) { clearInterval(state._health); state._health = null; } }
  function qualityToPipeline(q) {
    var map = { draft: "512", standard: "1024", high: "1024_cascade" };
    var want = map[q] || "1024";
    var allowed = (serverConfig && serverConfig.pipeline_types) || ["512", "1024", "1024_cascade"];
    if (allowed.indexOf(want) >= 0) return want;
    return (serverConfig && serverConfig.default_pipeline_type) || want;
  }

  // =========================================================================
  // Derived view values (ported from renderVals())
  // =========================================================================
  function vals() {
    var s = state;
    var navBase = "background:none;border:none;padding:8px 12px;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.22em;cursor:pointer;";
    var nav = function (k) {
      return navBase + (s.screen === k
        ? "color:#35F2E2;border-bottom:2px solid #35F2E2;text-shadow:0 0 12px rgba(53,242,226,.6)"
        : "color:#57c9bf;border-bottom:2px solid transparent");
    };
    var chipBase = "font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.22em;padding:9px 16px;white-space:nowrap;border:1px solid ";
    var stages = STAGE_NAMES.map(function (label, i) {
      var chip;
      if (i < s.stageIdx) chip = chipBase + "rgba(53,242,226,.8);background:rgba(53,242,226,.14);color:#7CFFF1";
      else if (i === s.stageIdx) chip = chipBase + "#35F2E2;background:#35F2E2;color:#031110;font-weight:600;animation:glowpulse 1.6s ease infinite";
      else chip = chipBase + "rgba(53,242,226,.18);color:#2e5d57";
      return { label: (i < s.stageIdx ? "✓ " : "") + label, chip: chip };
    });
    var dropBase = "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:clamp(40px,8vh,90px) 24px;cursor:pointer;text-align:center;border:1.5px dashed ";
    var dropStyle = s.dragOver
      ? dropBase + "#35F2E2;background:rgba(53,242,226,.09);box-shadow:0 0 40px rgba(53,242,226,.3), inset 0 0 40px rgba(53,242,226,.1)"
      : dropBase + "rgba(53,242,226,.4);background:rgba(4,28,26,.4)";
    return {
      nav: nav, stages: stages, dropStyle: dropStyle,
      userChip: (s.authName.trim() || "operator").toUpperCase(),
      queueLabel: queueLabelFor(),
      advArrow: s.advOpen ? "▾" : "▸",
      detailLabel: num(s.detail),
      progInt: Math.round(s.prog),
      barFill: "height:100%;width:" + s.prog + "%;background:repeating-linear-gradient(45deg,#35F2E2 0 10px,#14D9C9 10px 20px);box-shadow:0 0 18px rgba(53,242,226,.7);animation:stripes .7s linear infinite;transition:width .2s linear",
    };
  }

  // Header queue indicator — always tells the user how many jobs are in line.
  function queueLabelFor() {
    var s = state;
    var d = s.queueDepth || 0;
    if (s.phase === "busy") return s.busyReason === "full" ? "FULL · " + d + " WAITING" : "YOU'RE #" + (s.queuePos || 1);
    if (s.phase === "running" || s.phase === "warming") return d > 0 ? "1 RUNNING · " + d + " WAITING" : "YOUR JOB RUNNING";
    if (d > 0) return d + (d === 1 ? " JOB WAITING" : " JOBS WAITING");
    return "EMPTY";
  }

  // =========================================================================
  // Render
  // =========================================================================
  var LIGHT = !!window.__SOLIDIFY_LIGHT_FX;

  function render() {
    var html = "";
    if (state.screen === "login") html = renderLogin();
    else html = renderHeader() + renderScreen() + renderFooter();
    appEl.innerHTML = html;
    enhance();
    if (state.screen === "studio" && state.phase === "idle") refreshActivity();
    if (state.screen === "studio" && state.phase === "running") startFun(); else stopFun();
  }

  // Scroll-triggered reveals (GSAP ScrollTrigger). Content animates in as it
  // enters the viewport rather than only on screen mount — long pages (Home,
  // About) feel alive as you scroll. Falls back to plain visibility on reduced
  // motion / mobile. Re-armed on every render (innerHTML swap invalidates old
  // triggers), and ScrollTrigger.refresh() re-measures the new page height.
  function enhance() {
    var els = appEl.querySelectorAll("[data-reveal]");
    if (LIGHT || !window.gsap || !window.ScrollTrigger) {
      els.forEach(function (el) { el.style.opacity = ""; el.style.transform = ""; });
      return;
    }
    requestAnimationFrame(function () {
      window.ScrollTrigger.getAll().forEach(function (t) { t.kill(); });
      els.forEach(function (el) {
        window.gsap.set(el, { opacity: 0, y: 26 });
        window.ScrollTrigger.create({
          trigger: el, start: "top 90%", once: true,
          onEnter: function () { window.gsap.to(el, { opacity: 1, y: 0, duration: 0.9, ease: "power3.out" }); },
        });
      });
      window.ScrollTrigger.refresh();
      if (window.__lenis) window.__lenis.resize();
    });
  }

  function renderScreen() {
    if (state.screen === "home") return renderHome();
    if (state.screen === "about") return renderAbout();
    if (state.screen === "studio") return renderStudio();
    if (state.screen === "library") return renderLibrary();
    return "";
  }

  // ---- LOGIN ---------------------------------------------------------------
  function renderLogin() {
    var v = vals();
    return '' +
      '<div style="min-height:100vh;display:flex;flex-wrap:wrap;background:#010807">' +
        // left: specimen
        '<div style="flex:1.1 1 440px;min-height:60vh;position:relative;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:34px;padding:48px 24px;background-image:linear-gradient(rgba(53,242,226,.04) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.04) 1px, transparent 1px);background-size:44px 44px">' +
          '<div style="position:relative;width:min(400px,80%);aspect-ratio:843/1049;overflow:hidden;background:radial-gradient(circle at 50% 30%, #0a6f65 0%, #054a43 40%, #022a26 75%, #011a17 100%);border:1px solid rgba(53,242,226,.3);box-shadow:0 0 80px rgba(53,242,226,.28), inset 0 0 90px rgba(53,242,226,.12)">' +
            '<div style="position:absolute;inset:0;background-image:linear-gradient(rgba(53,242,226,.07) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.07) 1px, transparent 1px);background-size:34px 34px;pointer-events:none"></div>' +
            '<img class="biker-2d" src="' + ASSETS.biker + '" alt="operator specimen" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center top;mix-blend-mode:screen" />' +
            '<img class="biker-3d" src="' + ASSETS.biker3d + '" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center top;pointer-events:none" />' +
            '<div style="position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.05) 1px, transparent 1.5px);background-size:8px 8px;pointer-events:none"></div>' +
            '<div style="position:absolute;left:0;right:0;top:2%;height:2px;background:rgba(124,255,241,.95);box-shadow:0 0 20px rgba(124,255,241,1), 0 0 40px rgba(53,242,226,.8);animation:photoscan 6s ease-in-out infinite"></div>' +
            '<div style="position:absolute;left:6%;top:4%;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#7CFFF1;background:rgba(1,8,7,.72);padding:5px 9px;white-space:nowrap">SPECIMEN 042 · OPERATOR</div>' +
            '<div style="position:absolute;right:6%;bottom:4%;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#7CFFF1;background:rgba(1,8,7,.72);padding:5px 9px;white-space:nowrap">MESH LOCK ▮▮▮▮▯</div>' +
          '</div>' +
          '<div style="text-align:center">' +
            '<div style="font-size:clamp(40px,5.5vw,64px);font-weight:700;line-height:1;letter-spacing:-.01em"><glitch-text text="SOLIDIFY" speed="60"></glitch-text><span style="color:#35F2E2;animation:blinkc 1.1s step-end infinite">_</span></div>' +
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.3em;color:#57c9bf;margin-top:12px">ONE PHOTO IN → PRINT-READY MATTER OUT</div>' +
          '</div>' +
        '</div>' +
        // right: form
        '<div style="flex:1 1 400px;display:flex;align-items:center;justify-content:center;padding:48px 24px;position:relative;overflow:hidden;border-left:1px solid rgba(53,242,226,.14);background:#010807;background-image:linear-gradient(rgba(53,242,226,.06) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.06) 1px, transparent 1px);background-size:40px 40px">' +
          '<div style="position:absolute;inset:0;background:radial-gradient(circle at 60% 45%, rgba(20,217,201,.14), transparent 60%);pointer-events:none"></div>' +
          '<div style="position:absolute;left:0;right:0;height:40%;background:linear-gradient(180deg, transparent, rgba(53,242,226,.05), transparent);animation:scanmove 9s linear infinite;pointer-events:none"></div>' +
          '<form data-act="login" style="position:relative;width:min(400px,100%);display:flex;flex-direction:column;gap:22px;animation:rise .6s ease both;background:rgba(1,8,7,.88);border:1px solid rgba(53,242,226,.25);padding:34px 30px;backdrop-filter:blur(3px);box-shadow:0 0 70px rgba(1,8,7,.95)">' +
            '<div>' +
              '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf;margin-bottom:8px">RESTRICTED ACCESS · APPROVED OPERATORS ONLY</div>' +
              '<div style="font-size:28px;font-weight:600">Operator sign-in</div>' +
            '</div>' +
            (state.authError
              ? '<div style="border:1px solid rgba(255,84,104,.6);background:rgba(255,84,104,.08);padding:12px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#FF8A99;letter-spacing:.05em">✕ ACCESS DENIED — check your ID and key, or ask the lab admin to add you.</div>'
              : "") +
            '<label style="display:flex;flex-direction:column;gap:8px">' +
              '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.24em;color:#57c9bf">OPERATOR ID</span>' +
              '<input class="field" type="text" data-model="authName" value="' + esc(state.authName) + '" placeholder="j.haddad" autocomplete="username" style="background:rgba(4,28,26,.85);border:1px solid rgba(53,242,226,.28);color:#EAFDFB;padding:14px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:14px;outline:none" />' +
            '</label>' +
            '<label style="display:flex;flex-direction:column;gap:8px">' +
              '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.24em;color:#57c9bf">ACCESS KEY</span>' +
              '<input class="field" type="password" data-model="authKey" value="' + esc(state.authKey) + '" placeholder="••••••••" autocomplete="current-password" style="background:rgba(4,28,26,.85);border:1px solid rgba(53,242,226,.28);color:#EAFDFB;padding:14px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:14px;outline:none" />' +
            '</label>' +
            '<button class="h-primary" data-magnet type="submit" style="background:#14D9C9;color:#02201c;border:none;padding:16px 20px;font-size:15px;font-weight:700;letter-spacing:.12em;cursor:pointer;box-shadow:0 0 26px rgba(53,242,226,.35)">AUTHENTICATE →</button>' +
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#3f7a73;letter-spacing:.06em">' + (DEMO ? 'demo mode: any ID + key signs in · key "wrong" shows the denied state' : 'use the operator credentials issued by your lab admin') + '</div>' +
          '</form>' +
        '</div>' +
      '</div>';
  }

  // ---- HEADER / FOOTER -----------------------------------------------------
  function renderHeader() {
    var v = vals();
    return '' +
      '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px 18px;padding:14px clamp(16px,3vw,32px);border-bottom:1px solid rgba(53,242,226,.16);position:relative;z-index:5;background:rgba(1,8,7,.85);backdrop-filter:blur(4px)">' +
        '<div data-act="gotoHome" style="cursor:pointer;font-weight:700;font-size:18px;letter-spacing:.02em">SOLIDIFY<span style="color:#35F2E2;animation:blinkc 1.1s step-end infinite">_</span></div>' +
        '<div style="display:flex;gap:4px;margin-left:6px">' +
          '<button class="h-nav" data-act="gotoHome" style="' + v.nav("home") + '">HOME</button>' +
          '<button class="h-nav" data-act="gotoAbout" style="' + v.nav("about") + '">ABOUT</button>' +
          '<button class="h-nav" data-act="gotoStudio" style="' + v.nav("studio") + '">STUDIO</button>' +
          '<button class="h-nav" data-act="gotoLibrary" style="' + v.nav("library") + '">LIBRARY</button>' +
        '</div>' +
        '<div style="flex:1"></div>' +
        '<div style="display:flex;align-items:center;gap:8px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.18em;color:#6BFF2E;white-space:nowrap"><span style="width:8px;height:8px;background:#6BFF2E;border-radius:50%;box-shadow:0 0 10px #6BFF2E;animation:ledpulse 2.2s ease infinite"></span>MACHINE ONLINE</div>' +
        '<div style="display:flex;align-items:center;gap:8px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.18em;color:#57c9bf;white-space:nowrap">QUEUE <span id="hdr-queue">' + esc(v.queueLabel) + '</span></div>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;color:#EAFDFB;border:1px solid rgba(53,242,226,.3);padding:6px 12px">' + esc(v.userChip) + '</div>' +
        '<button class="h-ghost" data-act="logout" style="background:none;border:1px solid rgba(53,242,226,.3);color:#57c9bf;padding:6px 12px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.18em;cursor:pointer">LOGOUT</button>' +
      '</div>';
  }
  function renderFooter() {
    return '<div style="display:flex;flex-wrap:wrap;gap:8px 24px;justify-content:center;padding:14px 20px;border-top:1px solid rgba(53,242,226,.12);font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.2em;color:#2e5d57"><span>LAB MACHINE · SECURE TUNNEL</span><span>ONE JOB AT A TIME</span><span>ZERO INSTALL · ANY DEVICE</span></div>';
  }

  // ---- HOME ----------------------------------------------------------------
  function renderHome() {
    var ctaPrimary = 'data-magnet class="h-primary" style="background:#35F2E2;color:#02201c;border:none;padding:18px 36px;font-size:17px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 34px rgba(53,242,226,.45)"';
    var step = function (n, t, d) {
      return '<div class="h-card" data-spot style="background:#010807;padding:26px 24px"><div style="font-size:34px;font-weight:700;color:#35F2E2;text-shadow:0 0 20px rgba(53,242,226,.5)">' + n + '</div><div style="font-size:17px;font-weight:600;margin:12px 0 8px">' + t + '</div><div style="font-size:14px;color:#a8ded7;line-height:1.65">' + d + '</div></div>';
    };
    var stat = function (big, small, border) {
      return '<div style="flex:1 1 110px;padding:16px 20px' + (border ? ';border-right:1px solid rgba(53,242,226,.14)' : '') + '"><div style="font-size:26px;font-weight:700;color:#35F2E2">' + big + '</div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.2em;color:#57c9bf">' + small + '</div></div>';
    };
    return '<div>' +
      // HERO
      '<div style="position:relative;overflow:hidden;min-height:min(88vh,940px);display:flex;align-items:center">' +
        '<vanta-bg color="#35f2e2" background="#010807"></vanta-bg>' +
        '<div style="position:absolute;inset:0;background:linear-gradient(90deg, rgba(1,8,7,.94) 0%, rgba(1,8,7,.6) 55%, rgba(1,8,7,.25) 100%);pointer-events:none"></div>' +
        '<div style="position:absolute;left:0;right:0;height:32%;background:linear-gradient(180deg, transparent, rgba(53,242,226,.05), transparent);animation:scanmove 8s linear infinite;pointer-events:none"></div>' +
        '<div style="position:relative;width:100%;display:flex;flex-wrap:wrap;align-items:center;gap:40px;padding:clamp(48px,8vh,100px) clamp(20px,5vw,72px)">' +
          '<div style="flex:1.2 1 480px;max-width:760px">' +
            '<div data-reveal style="display:flex;align-items:center;gap:12px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.32em;color:#57c9bf;margin-bottom:22px"><span style="width:34px;height:1px;background:#35F2E2;box-shadow:0 0 8px #35F2E2"></span>IMAGE-TO-3D · REMOTE LAB PIPELINE</div>' +
            '<h1 data-reveal style="margin:0;font-size:clamp(46px,7.5vw,108px);line-height:.94;font-weight:700;letter-spacing:-.01em"><glitch-text text="PHOTO IN." speed="45"></glitch-text><br /><span style="color:#35F2E2;text-shadow:0 0 44px rgba(53,242,226,.6)"><glitch-text text="MATTER OUT." speed="45" delay="500"></glitch-text></span></h1>' +
            '<p data-reveal style="max-width:56ch;font-size:clamp(15px,1.6vw,18px);line-height:1.7;color:#a8ded7;margin:28px 0 12px">SOLIDIFY is the lab’s remote control for an image-to-3D AI pipeline. Upload one photo of any object — the lab’s GPU reconstructs it as a <span style="color:#EAFDFB">watertight, print-ready 3D model</span>. Preview it live, download STL or GLB, and print.</p>' +
            '<p data-reveal style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.1em;line-height:2;color:#3f7a73;margin:0 0 32px">NOTHING TO INSTALL · NO 3D SKILLS NEEDED · ANY DEVICE, ANYWHERE</p>' +
            '<div data-reveal style="display:flex;flex-wrap:wrap;gap:16px;align-items:center">' +
              '<button data-act="gotoStudio" ' + ctaPrimary + '>START A SCAN →</button>' +
              '<button class="h-ghost" data-act="gotoAbout" style="background:none;border:1px solid rgba(53,242,226,.45);color:#35F2E2;padding:17px 28px;font-family:\'IBM Plex Mono\',monospace;font-size:13px;letter-spacing:.18em;cursor:pointer;white-space:nowrap">HOW IT WORKS</button>' +
            '</div>' +
            '<div data-reveal style="display:flex;flex-wrap:wrap;margin-top:40px;border:1px solid rgba(53,242,226,.2);background:rgba(1,8,7,.65)">' +
              stat("1", "PHOTO NEEDED", true) + stat("~5<span style=\"font-size:.55em\"> MIN</span>", "PER JOB", true) + stat("STL·GLB", "OUTPUT FILES", true) + stat("0", "INSTALLS", false) +
            '</div>' +
          '</div>' +
          '<div data-reveal style="flex:1 1 340px;display:flex;justify-content:center">' +
            '<div data-lenis-prevent style="position:relative;width:min(460px,90vw);aspect-ratio:4/3;border:1px solid rgba(53,242,226,.28);box-shadow:0 0 90px rgba(53,242,226,.3);animation:floaty 7s ease-in-out infinite;background:radial-gradient(circle at 50% 45%, rgba(20,217,201,.14), transparent 65%)">' +
              '<model-preview mode="orbit" autorotate="true" src="' + ASSETS.stl + '" style="position:absolute;inset:0"></model-preview>' +
              '<div style="position:absolute;left:7%;top:6%;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#7CFFF1;background:rgba(1,8,7,.72);padding:5px 9px;white-space:nowrap;pointer-events:none">SPECIMEN 042 · LIVE 3D</div>' +
              '<div style="position:absolute;right:7%;bottom:6%;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#7CFFF1;background:rgba(1,8,7,.72);padding:5px 9px;white-space:nowrap;pointer-events:none">DRAG TO ROTATE · SCROLL TO ZOOM</div>' +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      // FIELD TEST
      '<div style="padding:clamp(48px,8vh,90px) clamp(20px,5vw,72px);border-top:1px solid rgba(53,242,226,.14);background-image:linear-gradient(rgba(53,242,226,.035) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.035) 1px, transparent 1px);background-size:44px 44px">' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.32em;color:#57c9bf;margin-bottom:10px">FIELD TEST · SPECIMEN 042</div>' +
        '<div style="font-size:clamp(28px,4vw,44px);font-weight:700;margin-bottom:8px"><glitch-text text="REAL RESULT FROM THE PIPELINE" speed="18"></glitch-text></div>' +
        '<p style="max-width:62ch;color:#a8ded7;font-size:15px;line-height:1.7;margin:0 0 36px">A single phone photo of a frog figurine — and the mesh the lab machine reconstructed from it, watertight and ready to slice. This is the entire input and the entire output.</p>' +
        '<div style="display:flex;flex-wrap:wrap;gap:26px;align-items:stretch">' +
          '<div data-spot style="flex:1 1 340px;position:relative;border:1px solid rgba(53,242,226,.3);background:radial-gradient(circle at 50% 45%, #0FD8C7 0%, #0a9a8d 60%, #067167 100%);min-height:300px;overflow:hidden">' +
            '<div style="position:absolute;inset:0;background-image:radial-gradient(rgba(2,32,28,.14) 1px, transparent 1.5px);background-size:8px 8px"></div>' +
            '<img src="' + ASSETS.frog + '" alt="2D input photo — frog figurine" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;padding:6%" />' +
            '<div style="position:absolute;left:12px;top:12px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.24em;color:#02201c;background:rgba(53,242,226,.95);padding:6px 10px;white-space:nowrap">INPUT · 2D PHOTO</div>' +
          '</div>' +
          '<div style="flex:0 0 auto;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;min-width:90px">' +
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.2em;color:#57c9bf">SOLIDIFY</div>' +
            '<div style="width:64px;height:10px;background:repeating-linear-gradient(90deg,#35F2E2 0 10px,transparent 10px 20px);animation:stripes .7s linear infinite;box-shadow:0 0 14px rgba(53,242,226,.5)"></div>' +
            '<div style="color:#35F2E2;font-size:22px">→</div>' +
          '</div>' +
          '<div data-spot style="flex:1 1 340px;position:relative;border:1px solid rgba(53,242,226,.3);background:#101413;min-height:300px;overflow:hidden">' +
            '<img src="' + ASSETS.mesh + '" alt="3D reconstructed mesh of the frog" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#101413" />' +
            '<div style="position:absolute;left:12px;top:12px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.24em;color:#02201c;background:#6BFF2E;padding:6px 10px;box-shadow:0 0 16px rgba(107,255,46,.5);white-space:nowrap">OUTPUT · PRINT-READY MESH</div>' +
            '<div style="position:absolute;right:12px;bottom:12px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.14em;color:#7CFFF1;background:rgba(1,8,7,.85);padding:6px 10px;border:1px solid rgba(53,242,226,.3);white-space:nowrap">WATERTIGHT ✓ · STL</div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      // PROTOCOL
      '<div style="padding:clamp(48px,8vh,90px) clamp(20px,5vw,72px);border-top:1px solid rgba(53,242,226,.14)">' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.32em;color:#57c9bf;margin-bottom:10px">PROTOCOL</div>' +
        '<div style="font-size:clamp(28px,4vw,44px);font-weight:700;margin-bottom:36px"><glitch-text text="FOUR STEPS. NO TRAINING." speed="18"></glitch-text></div>' +
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1px;background:rgba(53,242,226,.18);border:1px solid rgba(53,242,226,.18)">' +
          step("01", "UPLOAD A PHOTO", "One clear shot of the object, straight from your phone or laptop. That’s the whole input.") +
          step("02", "THE MACHINE WORKS", "Your photo travels through a secure tunnel to the lab GPU. One job at a time, a few minutes each.") +
          step("03", "WATCH IT FORM", "Live stages and a 3D visualization show your model materializing — waiting feels informative, not broken.") +
          step("04", "DOWNLOAD &amp; PRINT", "Rotate the finished model, check print diagnostics, grab the STL — it’s sealed and printable.") +
        '</div>' +
        '<div style="display:flex;justify-content:center;margin-top:44px"><button data-act="gotoStudio" data-magnet class="h-primary" style="background:#35F2E2;color:#02201c;border:none;padding:18px 40px;font-size:17px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 34px rgba(53,242,226,.45)">⬢ SOLIDIFY SOMETHING</button></div>' +
      '</div>' +
    '</div>';
  }

  // ---- ABOUT ---------------------------------------------------------------
  function renderAbout() {
    var card = function (tag, body) {
      return '<div data-reveal data-spot style="border:1px solid rgba(53,242,226,.22);padding:26px 24px;background:rgba(4,26,24,.35)"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.28em;color:#35F2E2;margin-bottom:14px">' + tag + '</div><div style="font-size:15px;color:#cdeee9;line-height:1.75">' + body + '</div></div>';
    };
    return '<div>' +
      '<div style="position:relative;overflow:hidden;padding:clamp(48px,9vh,100px) clamp(20px,5vw,72px);min-height:220px">' +
        '<letter-glitch colors="#04201c,#0b4a42,#15887b" speed="60"></letter-glitch>' +
        '<div style="position:relative">' +
          '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.32em;color:#57c9bf;margin-bottom:12px">DOSSIER</div>' +
          '<div style="font-size:clamp(36px,6vw,72px);font-weight:700;line-height:1"><glitch-text text="ABOUT SOLIDIFY" speed="35"></glitch-text></div>' +
        '</div>' +
      '</div>' +
      '<div style="padding:clamp(36px,6vh,70px) clamp(20px,5vw,72px);display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:26px;border-top:1px solid rgba(53,242,226,.14)">' +
        card("// THE IDEA", "SOLIDIFY is an interface — a remote control — for an image-to-3D AI model running on a single lab machine with a GPU. You bring a photo; the system returns a watertight, print-ready 3D model with a live preview before download. Open a link, sign in, use it from any device. Nothing to install, nothing to configure.") +
        card("// THE PROBLEM IT KILLS", "Turning a real object into a printable 3D model normally demands specialized software, technical setup, and know-how. SOLIDIFY collapses all of that into one upload — so professors and lab members produce print-ready files themselves, with zero tools and zero training.") +
        card("// THE MACHINE", "All computation runs on one lab Mac with a GPU — the site only sends your image and displays the result, over a secure tunnel. One job runs at a time and takes a few minutes. First job after a restart waits ~100 seconds while the model warms up.") +
        card("// WHO IT'S FOR", "Primary users are professors at the lab who want 3D-printable files from photos, fast, from their own devices. Access is restricted to an approved list managed by the lab admin. Students and other lab members: to be decided.") +
        card("// UNDER THE HOOD", '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;color:#8fd8cf;line-height:2.2">&gt; silhouette + depth extraction<br />&gt; volumetric field generation<br />&gt; marching cubes → mesh<br />&gt; decimation to target faces<br />&gt; watertight print-prep (hole sealing, wall thickening)<br />&gt; STL + GLB packaging</div>') +
        card("// ROADMAP", '<span style="color:#6BFF2E">▸</span> Proper accounts with an owner-managed allow-list<br /><span style="color:#57c9bf">▸</span> History / gallery of past generations<br /><span style="color:#57c9bf">▸</span> Print instructions &amp; slicer hand-off<br /><span style="color:#57c9bf">▸</span> English / Arabic support') +
      '</div>' +
      '<div style="display:flex;justify-content:center;padding:0 20px clamp(48px,8vh,80px)"><button data-act="gotoStudio" data-magnet class="h-primary" style="background:#35F2E2;color:#02201c;border:none;padding:18px 40px;font-size:17px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 34px rgba(53,242,226,.45)">START A SCAN →</button></div>' +
    '</div>';
  }

  // ---- LIBRARY -------------------------------------------------------------
  function renderLibrary() {
    var lib = loadLibrary();
    var header = '<div style="display:flex;flex-wrap:wrap;align-items:flex-end;gap:16px;margin-bottom:8px">' +
        '<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.32em;color:#57c9bf;margin-bottom:8px">ARCHIVE</div>' +
        '<div style="font-size:clamp(28px,4vw,44px);font-weight:700"><glitch-text text="SPECIMEN LIBRARY" speed="20"></glitch-text></div></div>' +
        '<div style="flex:1"></div>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.18em;color:#57c9bf">' + lib.length + ' MODEL' + (lib.length === 1 ? "" : "S") + ' ON THIS DEVICE</div>' +
      '</div>' +
      '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.06em;color:#3f7a73;line-height:1.7;margin-bottom:28px">Everything you’ve solidified, saved on this browser. Downloads pull from the lab machine while it still holds the job.</div>';

    var body;
    if (!lib.length) {
      body = '<div style="border:1px dashed rgba(53,242,226,.3);background:rgba(4,28,26,.35);padding:clamp(36px,7vh,80px) 24px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:16px">' +
        '<div style="width:56px;height:56px;border:1.5px solid rgba(53,242,226,.5);display:flex;align-items:center;justify-content:center;font-size:24px;color:#35F2E2">□</div>' +
        '<div style="font-size:20px;font-weight:600">No models yet.</div>' +
        '<div style="font-size:14px;color:#a8ded7;max-width:42ch;line-height:1.6">Solidify your first photo and it lands here — with a live 3D snapshot and one-click STL / GLB downloads.</div>' +
        '<button data-act="gotoStudio" data-magnet class="h-primary" style="margin-top:6px;background:#35F2E2;color:#02201c;border:none;padding:15px 30px;font-size:15px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 30px rgba(53,242,226,.4)">START A SCAN →</button>' +
      '</div>';
    } else {
      body = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px">' +
        lib.map(libCard).join("") + '</div>';
    }
    return '<div style="min-height:calc(100vh - 61px);padding:clamp(28px,5vh,56px) clamp(16px,4vw,48px);background-image:linear-gradient(rgba(53,242,226,.035) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.035) 1px, transparent 1px);background-size:44px 44px">' +
      header + body + '</div>';
  }

  function libCard(e) {
    var wt = e.watertight === false ? '<span style="color:#FFB454">NOT WATERTIGHT</span>'
      : (e.watertight ? '<span style="color:#6BFF2E">WATERTIGHT ✓</span>' : '');
    var thumb = e.thumb
      ? '<img src="' + esc(e.thumb) + '" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover" />'
      : '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#2e5d57;font-family:\'IBM Plex Mono\',monospace;font-size:11px">NO PREVIEW</div>';
    var stlBtn = e.stl ? '<button data-act="dl" data-path="' + esc(e.stl) + '" data-name="' + esc(e.stlName || "model.stl") + '" class="h-primary" style="flex:1;background:#35F2E2;color:#02201c;border:none;padding:10px;font-size:12px;font-weight:700;letter-spacing:.08em;cursor:pointer">↓ STL</button>' : "";
    var glbBtn = e.glb ? '<button data-act="dl" data-path="' + esc(e.glb) + '" data-name="' + esc(e.glbName || "model.glb") + '" class="h-ghost" style="flex:1;background:none;border:1px solid rgba(53,242,226,.45);color:#35F2E2;padding:10px;font-size:12px;font-weight:600;letter-spacing:.08em;cursor:pointer;font-family:\'IBM Plex Mono\',monospace">↓ GLB</button>' : "";
    return '<div data-spot style="border:1px solid rgba(53,242,226,.22);background:rgba(4,26,24,.35);display:flex;flex-direction:column;overflow:hidden">' +
        '<div style="position:relative;aspect-ratio:3/2;background:radial-gradient(circle at 50% 45%, rgba(20,217,201,.12), #02100e 70%);border-bottom:1px solid rgba(53,242,226,.18)">' + thumb +
          '<button data-act="libRemove" data-id="' + esc(e.id) + '" title="Remove from library" style="position:absolute;top:8px;right:8px;width:26px;height:26px;background:rgba(1,8,7,.7);border:1px solid rgba(255,84,104,.4);color:#FF8A99;font-size:13px;cursor:pointer;line-height:1">✕</button>' +
        '</div>' +
        '<div style="padding:14px 16px;display:flex;flex-direction:column;gap:10px">' +
          '<div style="font-size:16px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(e.name || "Model") + '</div>' +
          '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.08em;color:#57c9bf;display:flex;flex-wrap:wrap;gap:4px 12px">' +
            '<span>' + fmtDate(e.date) + '</span>' + (e.faces != null ? '<span>' + num(e.faces) + ' faces</span>' : "") +
          '</div>' +
          (wt ? '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.14em">' + wt + '</div>' : "") +
          '<div style="display:flex;gap:8px;margin-top:2px">' + stlBtn + glbBtn + '</div>' +
        '</div>' +
      '</div>';
  }

  // ---- STUDIO --------------------------------------------------------------
  function renderStudio() {
    var body = "";
    var p = state.phase;
    if (p === "idle") body = studioIdle();
    else if (p === "uploaded") body = studioUploaded();
    else if (p === "busy") body = studioBusy();
    else if (p === "warming") body = studioWarming();
    else if (p === "running") body = studioRunning();
    else if (p === "success") body = studioSuccess();
    else if (p === "error") body = studioError();
    return '<div style="min-height:calc(100vh - 61px);padding:clamp(20px,4vh,44px) clamp(16px,4vw,48px);position:relative;background-image:linear-gradient(rgba(53,242,226,.035) 1px, transparent 1px),linear-gradient(90deg, rgba(53,242,226,.035) 1px, transparent 1px);background-size:44px 44px">' + body + '</div>';
  }

  function studioIdle() {
    var v = vals();
    return '<div style="max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:26px;animation:rise .5s ease both">' +
      '<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf;margin-bottom:8px">NEW SCAN</div><div style="font-size:clamp(24px,3vw,32px);font-weight:600">Upload a photo of your object</div></div>' +
      (state.errMsg ? '<div style="border:1px solid rgba(255,180,84,.55);background:rgba(255,180,84,.07);padding:12px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#FFB454">△ ' + esc(state.errMsg) + '</div>' : "") +
      '<div data-act="browse" data-drop style="' + v.dropStyle + '">' +
        '<div style="width:56px;height:56px;border:1.5px solid #35F2E2;box-shadow:0 0 18px rgba(53,242,226,.45), inset 0 0 12px rgba(53,242,226,.15);display:flex;align-items:center;justify-content:center;font-size:26px;color:#35F2E2">↑</div>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;letter-spacing:.24em;color:#EAFDFB">DROP PHOTO HERE</div>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;color:#3f7a73">or click to browse · JPG / PNG / HEIC · max ' + state.maxUploadMB + ' MB</div>' +
      '</div>' +
      '<input type="file" accept="image/*" id="file-input" style="display:none" />' +
      '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.08em;color:#3f7a73;line-height:1.8">TIP — a single object on a plain background, evenly lit, gives the cleanest mesh.</div>' +
      // Live pipeline activity — every object currently being turned into 3D.
      '<div id="activity" style="margin-top:10px"></div>' +
    '</div>';
  }

  // Render the live queue/activity panel from GET /api/jobs into #activity.
  function activityRow(j) {
    var st = j.status;
    var done = st === "done", err = st === "error";
    var color = done ? "#6BFF2E" : err ? "#FF5468" : "#FFB454";
    var label = (STATUS[st] && STATUS[st].activity) || String(st).toUpperCase();
    var dot = (done || err)
      ? '<span style="width:8px;height:8px;border-radius:50%;background:' + color + '"></span>'
      : '<span style="width:8px;height:8px;border-radius:50%;background:' + color + ';box-shadow:0 0 8px ' + color + ';animation:ledpulse 1.4s ease infinite"></span>';
    return '<div style="display:flex;align-items:center;gap:12px;padding:9px 12px;border:1px solid rgba(53,242,226,.16);background:rgba(2,13,12,.55)">' +
      dot +
      '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;color:#EAFDFB">JOB ' + esc(shortId(j.job_id)) + '</span>' +
      '<span style="flex:1"></span>' +
      '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.2em;color:' + color + '">' + label + '</span>' +
    '</div>';
  }
  // Throttled so idle-screen re-renders (drag-over toggles, validation errors)
  // don't each fire a GET /api/jobs; also refreshes the header queue count from
  // the same payload so the studio-idle tick needs only this one request.
  var _activityAt = 0, _activityBusy = false;
  async function refreshActivity(now) {
    var el = appEl.querySelector("#activity");
    if (!el) return;
    now = now || (window.performance ? performance.now() : 0);
    if (_activityBusy || now - _activityAt < 1500) return;
    _activityBusy = true; _activityAt = now;
    var jobs;
    try { jobs = (await (await api("/api/jobs")).json()).jobs || []; }
    catch (e) { return; }
    finally { _activityBusy = false; }
    var live = jobs.filter(function (j) { return STATUS[j.status] && STATUS[j.status].active; });
    state.queueDepth = live.length; patchQueueHeader();
    var head = '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf;margin:8px 0 10px">PIPELINE NOW · ' + live.length + ' IN QUEUE</div>';
    if (!live.length) {
      el.innerHTML = head + '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.06em;color:#3f7a73">Machine idle — your job would start immediately.</div>';
    } else {
      el.innerHTML = head + '<div style="display:flex;flex-direction:column;gap:6px">' + live.map(activityRow).join("") + '</div>';
    }
  }

  function studioUploaded() {
    return '<div style="max-width:1080px;margin:0 auto;display:flex;flex-wrap:wrap;gap:32px;animation:rise .5s ease both">' +
      '<div style="flex:1 1 340px;display:flex;flex-direction:column;gap:14px">' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf">SOURCE PHOTO</div>' +
        '<div style="position:relative;border:1px solid rgba(53,242,226,.35);overflow:hidden;background:#020d0c">' +
          '<div role="img" aria-label="uploaded object" style="width:100%;height:380px;background-image:' + (state.photoSrc ? "url(" + state.photoSrc + ")" : "none") + ';background-size:contain;background-repeat:no-repeat;background-position:center"></div>' +
          '<div style="position:absolute;left:0;right:0;height:3px;background:rgba(53,242,226,.85);box-shadow:0 0 18px rgba(53,242,226,.9);animation:photoscan 4.5s ease-in-out infinite"></div>' +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:12px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#57c9bf;letter-spacing:.06em">' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(state.photoName) + '</span>' +
          '<span>' + esc(state.photoSize) + '</span>' +
          '<button class="h-remove" data-act="clearPhoto" style="background:none;border:1px solid rgba(255,84,104,.4);color:#FF8A99;padding:4px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.14em;cursor:pointer">REMOVE</button>' +
        '</div>' +
      '</div>' +
      '<div style="flex:1 1 380px;display:flex;flex-direction:column;gap:22px">' +
        '<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf;margin-bottom:8px">READY TO GENERATE</div><div style="font-size:24px;font-weight:600">Defaults are dialed in — just hit generate.</div></div>' +
        '<button data-act="generate" data-magnet class="h-primary" style="background:#35F2E2;color:#02201c;border:none;padding:20px 28px;font-size:18px;font-weight:700;letter-spacing:.12em;cursor:pointer;box-shadow:0 0 34px rgba(53,242,226,.45)">⬢ GENERATE 3D MODEL</button>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;color:#3f7a73">TAKES A FEW MINUTES · ONE JOB AT A TIME ON THE LAB MACHINE</div>' +
        studioAdvanced() +
      '</div>' +
    '</div>';
  }

  function studioAdvanced() {
    var v = vals();
    var open = state.advOpen ? (
      '<div style="padding:20px 18px;display:flex;flex-direction:column;gap:20px;border-top:1px solid rgba(53,242,226,.14)">' +
        '<label style="display:flex;flex-direction:column;gap:8px"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.22em;color:#57c9bf">RANDOM SEED</span>' +
          '<input class="field" type="text" data-model="seed" value="' + esc(state.seed) + '" placeholder="random" style="background:rgba(4,28,26,.85);border:1px solid rgba(53,242,226,.25);color:#EAFDFB;padding:10px 12px;font-family:\'IBM Plex Mono\',monospace;font-size:13px;outline:none;max-width:200px" /></label>' +
        '<label style="display:flex;flex-direction:column;gap:8px"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.22em;color:#57c9bf">MESH DETAIL — <span id="detail-label" style="color:#35F2E2">' + v.detailLabel + '</span> FACES</span>' +
          '<input type="range" min="20000" max="200000" step="5000" value="' + state.detail + '" data-model="detail" style="accent-color:#35F2E2;width:100%" /></label>' +
        '<label style="display:flex;flex-direction:column;gap:8px"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.22em;color:#57c9bf">PIPELINE QUALITY</span>' +
          '<select data-model="quality" style="background:rgba(4,28,26,.85);border:1px solid rgba(53,242,226,.25);color:#EAFDFB;padding:10px 12px;font-family:\'IBM Plex Mono\',monospace;font-size:13px;outline:none;max-width:240px">' +
            '<option value="draft"' + (state.quality === "draft" ? " selected" : "") + '>Draft — fastest</option>' +
            '<option value="standard"' + (state.quality === "standard" ? " selected" : "") + '>Standard — recommended</option>' +
            '<option value="high"' + (state.quality === "high" ? " selected" : "") + '>High — slowest, finest</option>' +
          '</select></label>' +
        '<label style="display:flex;align-items:center;gap:12px;cursor:pointer"><input type="checkbox" data-model="skipPrep"' + (state.skipPrep ? " checked" : "") + ' style="accent-color:#35F2E2;width:16px;height:16px" /><span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.12em;color:#a8ded7">SKIP PRINT-PREP <span style="color:#3f7a73">(raw geometry only — not watertight)</span></span></label>' +
      '</div>'
    ) : "";
    return '<div style="border:1px solid rgba(53,242,226,.2)">' +
      '<button class="h-nav" data-act="toggleAdv" style="width:100%;display:flex;align-items:center;gap:10px;background:rgba(4,28,26,.6);border:none;color:#57c9bf;padding:14px 18px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.24em;cursor:pointer;text-align:left"><span>' + v.advArrow + '</span><span>ADVANCED PARAMETERS</span><span style="flex:1"></span><span style="color:#3f7a73;letter-spacing:.06em">optional</span></button>' +
      open +
    '</div>';
  }

  function studioBusy() {
    var full = state.busyReason === "full";
    var head = full ? "◐ LAB AT CAPACITY" : "◐ MACHINE BUSY";
    var title = full ? "The queue is full right now." : "Someone else’s model is moving through the pipeline.";
    var body = full
      ? "The lab machine can only hold so many jobs in line. Give it a moment to clear, then send yours again — your photo and settings are still here."
      : "The lab machine runs one job at a time. Yours is safely queued and will start automatically — no need to do anything.";
    var cancelBtn = '<button data-act="cancelQueue" class="h-ghost" style="background:none;border:1px solid rgba(255,84,104,.4);color:#FF8A99;padding:12px 22px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.16em;cursor:pointer">✕ CANCEL — LEAVE QUEUE</button>';
    var meter = full
      ? '<div id="busy-meter" style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.14em;color:#FFB454;margin-top:2px">' + (state.queuePos || 0) + ' JOB(S) AHEAD · WAITING FOR A FREE SLOT…</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px">' +
          '<button data-act="generate" class="h-primary" style="background:#FFB454;color:#241400;border:none;padding:14px 24px;font-size:14px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 22px rgba(255,180,84,.35)">↻ TRY NOW</button>' +
          cancelBtn +
        '</div>'
      : '<div style="display:flex;align-items:baseline;gap:14px;margin-top:6px"><span id="busy-pos" style="font-size:56px;font-weight:700;color:#FFB454">' + (state.queuePos || 1) + '</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.2em;color:#57c9bf">POSITION IN QUEUE</span></div>' +
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;color:#3f7a73">Your model starts automatically the moment it reaches the front.</div>' +
        '<div style="margin-top:6px">' + cancelBtn + '</div>';
    return '<div style="max-width:640px;margin:6vh auto 0;border:1px solid rgba(255,180,84,.5);background:rgba(255,180,84,.05);padding:clamp(24px,4vw,44px);display:flex;flex-direction:column;gap:16px;animation:rise .5s ease both">' +
      '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#FFB454">' + head + '</div>' +
      '<div style="font-size:26px;font-weight:600">' + title + '</div>' +
      '<div style="font-size:15px;color:#a8ded7;line-height:1.7">' + body + '</div>' +
      meter +
    '</div>';
  }
  // Update the live position number without tearing down the busy screen.
  function patchBusy() {
    var el = appEl.querySelector("#busy-pos");
    if (el) el.textContent = state.queuePos || 1;
  }

  function studioWarming() {
    return '<div style="max-width:640px;margin:6vh auto 0;border:1px solid rgba(53,242,226,.35);background:rgba(53,242,226,.04);padding:clamp(24px,4vw,44px);display:flex;flex-direction:column;gap:16px;animation:rise .5s ease both">' +
      '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#35F2E2">◌ WARMING UP</div>' +
      '<div style="font-size:26px;font-weight:600">First run of the day — loading the AI model.</div>' +
      '<div style="font-size:15px;color:#a8ded7;line-height:1.7">The pipeline loads into GPU memory the first time after a restart (about 100 seconds). Your job starts the moment it’s ready.</div>' +
      '<div style="height:8px;border:1px solid rgba(53,242,226,.35);overflow:hidden"><div style="height:100%;width:40%;background:repeating-linear-gradient(45deg,#35F2E2 0 10px,rgba(53,242,226,.35) 10px 20px);animation:stripes .8s linear infinite"></div></div>' +
    '</div>';
  }

  function studioRunning() {
    var v = vals();
    return '<div style="max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:28px;animation:rise .5s ease both">' +
      '<div style="display:flex;flex-wrap:wrap;align-items:flex-end;gap:18px">' +
        '<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#57c9bf;margin-bottom:8px">JOB ' + esc(state.jobId) + ' · RUNNING ON LAB MACHINE</div><div style="font-size:clamp(24px,3vw,30px);font-weight:600">Solidifying your photo…</div>' +
          '<div id="fun-word" style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;letter-spacing:.16em;color:#35F2E2;margin-top:10px">' + funWordHTML() + '</div></div>' +
        '<div style="flex:1"></div>' +
        '<div style="font-size:clamp(44px,6vw,64px);font-weight:700;color:#35F2E2;text-shadow:0 0 28px rgba(53,242,226,.5);line-height:1"><span id="prog-int">' + v.progInt + '</span><span style="font-size:.5em">%</span></div>' +
      '</div>' +
      '<div id="stage-row" style="display:flex;flex-wrap:wrap;gap:10px">' + stageChips(v) + '</div>' +
      '<div style="height:10px;border:1px solid rgba(53,242,226,.35);overflow:hidden;background:rgba(4,28,26,.6)"><div id="bar-fill" style="' + v.barFill + '"></div></div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:28px;align-items:stretch">' +
        '<div data-lenis-prevent style="flex:1 1 380px;min-height:320px;border:1px solid rgba(53,242,226,.22);background:radial-gradient(circle at 50% 45%, rgba(20,217,201,.09), transparent 65%)">' +
          '<model-preview data-mp mode="forming" progress="' + v.progInt + '" autorotate="true" style="width:100%;height:100%;min-height:320px"></model-preview>' +
        '</div>' +
        '<div style="flex:1 1 300px;display:flex;flex-direction:column;gap:12px">' +
          '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#57c9bf">LIVE FEED</div>' +
          '<div id="log-feed" style="flex:1;border:1px solid rgba(53,242,226,.2);background:rgba(2,13,12,.85);padding:16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;line-height:2;color:#6fd8cc;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end">' + logLinesHTML() + '</div>' +
          '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.1em;color:#3f7a73">A REAL JOB TAKES A FEW MINUTES — YOU CAN LEAVE THIS TAB OPEN.</div>' +
        '</div>' +
      '</div>' +
    '</div>';
  }
  function stageChips(v) {
    return v.stages.map(function (st) { return '<div style="' + st.chip + '">' + st.label + '</div>'; }).join("");
  }
  function logLinesHTML() {
    return state.log.map(function (ln) { return '<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(ln) + '</div>'; }).join("") +
      '<div style="color:#35F2E2">▮<span style="animation:blinkc 1s step-end infinite">▮</span></div>';
  }

  function studioSuccess() {
    var infoRows = (state.infoRows || []).map(function (r) {
      return '<div style="display:flex;justify-content:space-between;gap:12px;font-size:14px"><span style="color:#7fb8b0;white-space:nowrap">' + esc(r[0]) + '</span><span style="font-family:\'IBM Plex Mono\',monospace;color:#EAFDFB;white-space:nowrap;text-align:right">' + esc(r[1]) + '</span></div>';
    }).join("");
    var diagRows = (state.diagRows || []).map(function (d) {
      var style = d[1] === "ok" ? "color:#6BFF2E;font-weight:700" : "color:#FFB454;font-weight:700";
      return '<div style="display:flex;gap:10px;font-size:13px;line-height:1.5"><span style="' + style + '">' + d[0] + '</span><span style="color:#a8ded7">' + esc(d[2]) + '</span></div>';
    }).join("");
    var srcAttr = state.previewSrc ? ' src="' + esc(state.previewSrc) + '"' : "";
    return '<div style="max-width:1120px;margin:0 auto;display:flex;flex-direction:column;gap:24px;animation:rise .6s ease both">' +
      '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:14px">' +
        '<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#6BFF2E;margin-bottom:8px">✓ JOB ' + esc(state.jobId) + ' COMPLETE</div><div style="font-size:clamp(24px,3vw,32px);font-weight:600">Your model is ready.</div></div>' +
        '<div style="flex:1"></div>' +
        '<button class="h-ghost" data-act="startOver" style="background:none;border:1px solid rgba(53,242,226,.4);color:#35F2E2;padding:12px 20px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.18em;cursor:pointer;white-space:nowrap">+ NEW SCAN</button>' +
      '</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:26px;align-items:stretch">' +
        '<div data-lenis-prevent style="flex:2 1 440px;min-height:420px;position:relative;border:1px solid rgba(53,242,226,.3);background:radial-gradient(circle at 50% 45%, rgba(20,217,201,.1), transparent 65%)">' +
          '<model-preview data-mp mode="orbit" autorotate="true"' + srcAttr + ' style="width:100%;height:100%;min-height:420px"></model-preview>' +
          '<div style="position:absolute;left:14px;top:12px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.24em;color:#57c9bf;pointer-events:none">LIVE PREVIEW — DRAG TO ROTATE · SCROLL TO ZOOM</div>' +
        '</div>' +
        '<div style="flex:1 1 320px;display:flex;flex-direction:column;gap:18px">' +
          '<div data-spot style="border:1px solid rgba(53,242,226,.22);padding:18px 20px"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#57c9bf;margin-bottom:14px">MODEL INFO</div><div style="display:flex;flex-direction:column;gap:10px">' + infoRows + '</div></div>' +
          '<div data-spot style="border:1px solid rgba(53,242,226,.22);padding:18px 20px"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:.26em;color:#57c9bf;margin-bottom:14px">PRINT DIAGNOSTICS</div><div style="display:flex;flex-direction:column;gap:10px">' + diagRows + '</div></div>' +
          '<button data-act="downloadSTL" data-magnet class="h-primary" style="background:#35F2E2;color:#02201c;border:none;padding:16px 22px;font-size:15px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 26px rgba(53,242,226,.4)">↓ PRINT-READY .STL</button>' +
          '<button data-act="downloadGLB" class="h-ghost" style="background:none;border:1px solid rgba(53,242,226,.45);color:#35F2E2;padding:14px 22px;font-size:13px;font-weight:600;letter-spacing:.12em;cursor:pointer;font-family:\'IBM Plex Mono\',monospace">↓ RAW GEOMETRY .GLB</button>' +
        '</div>' +
      '</div>' +
    '</div>';
  }

  function studioError() {
    return '<div style="max-width:640px;margin:6vh auto 0;border:1px solid rgba(255,84,104,.55);background:rgba(255,84,104,.05);padding:clamp(24px,4vw,44px);display:flex;flex-direction:column;gap:18px;animation:rise .5s ease both">' +
      '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.3em;color:#FF5468">✕ GENERATION FAILED</div>' +
      '<div style="font-size:26px;font-weight:600">That one didn’t make it through.</div>' +
      '<div style="font-size:15px;color:#a8ded7;line-height:1.7">' + esc(state.errMsg) + '</div>' +
      (state.errHint ? '<pre style="white-space:pre-wrap;word-break:break-word;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#8fd8cf;background:rgba(2,13,12,.7);border:1px solid rgba(53,242,226,.18);padding:12px 14px;margin:0">' + esc(state.errHint) + '</pre>' : "") +
      '<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:4px">' +
        '<button data-act="retry" class="h-danger" style="background:#FF5468;color:#1a0508;border:none;padding:15px 26px;font-size:15px;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 24px rgba(255,84,104,.4)">↻ RETRY SAME SETTINGS</button>' +
        '<button data-act="startOver" class="h-ghost" style="background:none;border:1px solid rgba(53,242,226,.4);color:#35F2E2;padding:15px 22px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:.16em;cursor:pointer">START OVER</button>' +
      '</div>' +
    '</div>';
  }

  // Patch the running screen in place (avoids tearing down the <model-preview> canvas).
  function patchRunning() {
    var v = vals();
    var pi = appEl.querySelector("#prog-int"); if (pi) pi.textContent = v.progInt;
    var bf = appEl.querySelector("#bar-fill"); if (bf) bf.style.width = state.prog + "%";
    var sr = appEl.querySelector("#stage-row"); if (sr) sr.innerHTML = stageChips(v);
    var lf = appEl.querySelector("#log-feed"); if (lf) lf.innerHTML = logLinesHTML();
    var mp = appEl.querySelector("[data-mp]"); if (mp) mp.setAttribute("progress", v.progInt);
  }

  // =========================================================================
  // Actions
  // =========================================================================
  function go(screen) { state.screen = screen; render(); }

  async function doLogin() {
    var id = state.authName.trim(), key = state.authKey.trim();
    if (!id || !key) { state.authError = true; render(); return; }
    if (DEMO) {
      if (key.toLowerCase() === "wrong") { state.authError = true; render(); return; }
      state.authError = false; state.screen = "home"; render(); startHealthPoll(); return;
    }
    // Attach the credential optimistically so the probe (and everything after)
    // goes through the same api()/authHeaders path; clear it again on a 401.
    AUTH = "Basic " + btoa(id + ":" + key);
    try {
      var r = await api("/api/health");
      if (r.status === 401) { AUTH = null; state.authError = true; render(); return; }
      state.authError = false; state.screen = "home";
      await loadConfig();
      render();
      startHealthPoll();
    } catch (e) { AUTH = null; state.authError = true; render(); }
  }

  // Stop every timer + revoke blob URLs + clear the in-flight job/result state.
  // Leaves the uploaded photo, screen, and auth untouched — callers set those.
  function resetJob() {
    stopPolling(); stopAutoRetry(); stopSim(); stopFun();
    revokeObjectUrls();
    localStorage.removeItem(LS_KEY);
    Object.assign(state, {
      log: [], prog: 0, stageIdx: -1, errMsg: "", errHint: "", busyReason: "", jobId: "",
      previewSrc: null, previewPath: null, libDraft: null,
      dlSTL: null, dlSTLName: "", dlGLB: null, dlGLBName: "",
    });
  }

  function logout() {
    resetJob(); stopHealthPoll();
    AUTH = null;
    Object.assign(state, { screen: "login", authKey: "", authError: false, phase: "idle", photoSrc: null, photoFile: null });
    render();
  }

  function acceptFile(file) {
    if (!file) return;
    if (!file.type || file.type.indexOf("image/") !== 0) { state.errMsg = "That file isn’t an image — try a JPG or PNG photo of the object."; render(); return; }
    if (file.size > state.maxUploadMB * 1024 * 1024) { state.errMsg = "That image is over the " + state.maxUploadMB + " MB limit — a phone photo works fine."; render(); return; }
    var rd = new FileReader();
    rd.onload = function () {
      state.photoSrc = rd.result; state.photoName = file.name; state.photoFile = file;
      state.photoSize = (file.size / 1048576).toFixed(1) + " MB"; state.phase = "uploaded"; state.errMsg = "";
      render();
    };
    rd.readAsDataURL(file);
  }

  function clearPhoto() { state.photoSrc = null; state.photoFile = null; state.phase = "idle"; render(); }

  async function generate() {
    if (DEMO) return simGenerate();
    if (!state.photoFile) return;
    state.log = []; state.prog = 0; state.stageIdx = -1; state.errMsg = ""; state.errHint = "";
    var fd = new FormData();
    fd.append("image", state.photoFile);
    if (state.seed !== "") fd.append("seed", state.seed);
    fd.append("target_faces", String(state.detail));
    fd.append("pipeline_type", qualityToPipeline(state.quality));
    fd.append("skip_printable", state.skipPrep ? "1" : "0");
    try {
      var r = await api("/api/jobs", { method: "POST", body: fd });
      if (r.status === 429) {
        // Queue is FULL — the job was rejected, not accepted. Keep the photo and
        // AUTO-RETRY: poll for a free slot and resubmit as soon as one opens, so
        // the user's turn advances to generation on its own. Cancel stops this.
        await refreshHealth();
        state.busyReason = "full"; state.queuePos = state.queueDepth || 0; state.phase = "busy";
        render();
        scheduleAutoRetry();
        return;
      }
      var d = await r.json();
      if (!r.ok) throw new Error(d.error || "Upload failed.");
      stopAutoRetry();
      state.jobId = shortId(d.job_id) || "JOB";
      state.busyReason = "";
      localStorage.setItem(LS_KEY, d.job_id);
      state.phase = "running"; state.stageIdx = 0; state.prog = 4;
      pushLog("> job " + state.jobId + " accepted · tunnel secure");
      render();
      startPolling(d.job_id);
    } catch (e) {
      state.phase = "error"; state.errMsg = String(e.message || e); state.errHint = ""; render();
    }
  }

  function retry() { state.phase = "uploaded"; generate(); }
  function startOver() {
    resetJob();
    state.phase = "idle"; state.photoSrc = null; state.photoFile = null;
    render();
  }

  // Auto-retry while the queue is full: resubmit periodically so the user's turn
  // advances to generation on its own the moment a slot frees.
  function scheduleAutoRetry() {
    stopAutoRetry();
    state._retry = setInterval(function () {
      if (state.phase !== "busy" || state.busyReason !== "full") { stopAutoRetry(); return; }
      generate(); // 429 again → re-schedules; 202 → clears + goes to running
    }, 6000);
  }
  function stopAutoRetry() { if (state._retry) { clearInterval(state._retry); state._retry = null; } }
  function stopSim() { if (state._sim) { clearInterval(state._sim); state._sim = null; } }

  // Rotating playful status word while a job runs. Driven by render() (start on
  // the running screen, stop otherwise); patches the live element in place.
  function funWordHTML() {
    return '✻ ' + esc(FUN_WORDS[state.funIdx % FUN_WORDS.length]) +
      '<span style="animation:blinkc 1s step-end infinite">…</span>';
  }
  function patchFun() { var el = appEl.querySelector("#fun-word"); if (el) el.innerHTML = funWordHTML(); }
  function startFun() {
    if (state._fun) return;
    state.funIdx = 0; // begin each run at the first word ("Cooking")
    patchFun();
    state._fun = setInterval(function () { state.funIdx++; patchFun(); }, 2200);
  }
  function stopFun() { if (state._fun) { clearInterval(state._fun); state._fun = null; } }

  // Cancel: stop waiting in the queue and return to the ready screen (photo kept).
  // NOTE: the backend has no job-cancel endpoint, so an already-accepted job keeps
  // running server-side; this stops the client from waiting on it.
  function cancelQueue() {
    resetJob();
    state.phase = state.photoFile ? "uploaded" : "idle";
    render();
  }

  function triggerDownload(url, name) {
    var a = document.createElement("a");
    a.href = url; a.download = name || ""; document.body.appendChild(a); a.click(); a.remove();
  }
  // Fetch the file with the auth header (a plain <a> GET wouldn't send it), then
  // save the resulting blob. Revoke shortly after so memory doesn't leak.
  async function downloadFile(path, name, el) {
    var label = el ? el.textContent : "";
    if (el) { el.dataset.busy = "1"; el.textContent = "PREPARING…"; }
    try {
      var url = await apiFileURL(path);
      triggerDownload(url, name);
      setTimeout(function () { try { URL.revokeObjectURL(url); } catch (e) {} }, 8000);
    } catch (e) {
      if (el) el.textContent = "DOWNLOAD FAILED — RETRY";
      return;
    } finally {
      if (el && el.dataset.busy) { el.textContent = label; delete el.dataset.busy; }
    }
  }
  function downloadResult(kind, el) {
    var path = kind === "STL" ? state.dlSTL : state.dlGLB;
    var name = kind === "STL" ? state.dlSTLName : state.dlGLBName;
    if (path) return downloadFile(path, name, el);
    if (DEMO) return triggerDownload("data:text/plain," + encodeURIComponent(demoSTL()),
      "solidify_demo_" + (kind === "STL" ? "print.stl" : "raw.glb"));
  }

  // =========================================================================
  // Polling (real backend)
  // =========================================================================
  function startPolling(id) { stopPolling(); poll(id); state._poll = setInterval(function () { poll(id); }, 2000); }
  function stopPolling() { if (state._poll) { clearInterval(state._poll); state._poll = null; } }

  async function poll(id) {
    var job;
    try {
      var r = await api("/api/jobs/" + id);
      if (r.status === 404) { stopPolling(); localStorage.removeItem(LS_KEY); state.phase = "idle"; render(); return; }
      job = await r.json();
    } catch (e) { return; /* transient — keep polling */ }
    applyJob(id, job);
  }

  function applyJob(id, job) {
    var st = job.status;
    if (st === "error") {
      stopPolling(); localStorage.removeItem(LS_KEY);
      var e = job.error || {};
      state.phase = "error"; state.errMsg = e.message || "Generation failed."; state.errHint = e.hint || "";
      render(); return;
    }
    if (st === "done") {
      stopPolling(); localStorage.removeItem(LS_KEY);
      buildSuccess(id, job); state.phase = "success"; render();
      loadPreview(); // fetch the GLB with auth, then feed the live viewer
      return;
    }
    if (st === "loading_model") {
      if (state.phase !== "warming") { pushLog(stageLog(st)); state.phase = "warming"; render(); }
      return;
    }
    if (st === "queued") {
      // On the single-worker backend a job only lingers in `queued` while the
      // machine is busy with an earlier job — so this is the "someone else is
      // running, you're in line" case. Show the position screen; keep polling.
      var wasQueued = state.phase === "busy" && state.busyReason === "queued";
      state.phase = "busy"; state.busyReason = "queued";
      state.queuePos = Math.max(1, state.queueDepth || 1);
      if (wasQueued) patchBusy(); else render();
      refreshHealth(); // freshen queueDepth + header for the next tick
      return;
    }
    // preprocessing / generating / making_printable → running
    var idx = STATUS[st] && STATUS[st].idx != null ? STATUS[st].idx : 0;
    var wasRunning = state.phase === "running";
    state.phase = "running"; state.busyReason = "";
    if (!state.jobId) state.jobId = shortId(id);
    if (idx !== state.stageIdx) pushLog(stageLog(st));
    state.stageIdx = idx;
    state.prog = progFor(idx, job.elapsed_seconds || 0);
    if (!wasRunning) render(); else patchRunning();
  }

  function progFor(idx, elapsed) {
    var base = [8, 28, 55, 92][idx] || 5;
    if (idx === 2) return Math.min(88, 55 + elapsed * 0.6); // grow through the long generating stage
    return base;
  }
  function pushLog(line) { state.log = state.log.concat([line]).slice(-7); }

  function buildSuccess(id, job) {
    var result = job.result || {}; var files = result.files || []; var base = "/api/jobs/" + id + "/files/";
    // Preview = the printable GLB (or raw GLB fallback), loaded via auth blob after render.
    state.previewPath = result.preview_filename ? base + result.preview_filename : null;
    state.previewSrc = null;
    var stl = files.filter(function (f) { return /\.stl$/i.test(f.filename); })[0];
    var glb = files.filter(function (f) { return /\.glb$/i.test(f.filename); })[0];
    // dlSTL/dlGLB hold API paths; the download handlers fetch them with auth on click.
    state.dlSTL = stl ? base + stl.filename : null; state.dlSTLName = stl ? stl.filename : "";
    state.dlGLB = glb ? base + glb.filename : null; state.dlGLBName = glb ? glb.filename : "";

    var ms = result.mesh_stats || {}; var rows = [];
    if (ms.vertices != null) rows.push(["Vertices", num(ms.vertices)]);
    if (ms.faces != null) rows.push(["Faces", num(ms.faces)]);
    if (result.dimensions_mm) rows.push(["Size", result.dimensions_mm]);
    rows.push(["Watertight", result.watertight === false ? "NO (prep skipped)" : (result.watertight ? "YES" : "—")]);
    var exts = files.map(function (f) { return String(f.filename).split(".").pop().toUpperCase(); });
    rows.push(["Files", exts.length ? exts.join(" · ") : "STL · GLB"]);
    state.infoRows = rows;

    var d = job.diagnostics || {}; var diag = [];
    if (result.watertight === false || state.skipPrep) diag.push(["△", "warn", "Print-prep was skipped — mesh is not watertight. Re-run with prep on before printing."]);
    else diag.push(["✓", "ok", "Watertight — sealed and ready to slice."]);
    if (d.thin_wall_warnings != null) {
      if (d.thin_wall_warnings > 0) diag.push(["△", "warn", d.thin_wall_warnings + " thin-wall warning(s)" + (d.thin_wall_threshold_mm ? " (below " + d.thin_wall_threshold_mm + " mm)" : "") + "."]);
      else diag.push(["✓", "ok", "Wall thickness within safe limits."]);
    }
    if (d.overhang_angle_deg != null) diag.push(["△", "warn", "Overhangs up to " + d.overhang_angle_deg + "° — add supports in your slicer."]);
    state.diagRows = diag.length ? diag : [["✓", "ok", "Model ready."]];

    // Draft a Library entry; the 3D snapshot is attached once the viewer renders.
    state.libDraft = {
      id: id, jobId: state.jobId,
      name: baseName(state.photoName) || ("Specimen " + state.jobId),
      date: new Date().toISOString(),
      faces: ms.faces, vertices: ms.vertices, watertight: result.watertight,
      stl: state.dlSTL, stlName: state.dlSTLName, glb: state.dlGLB, glbName: state.dlGLBName,
    };
  }

  // Wait for the viewer's model, let a few frames render, then snapshot it into
  // the Library. Shared by the real and demo success paths so the capture timing
  // lives in one place.
  async function captureToLibrary(mp, draft, timeoutMs) {
    if (!mp || !mp.whenLoaded || !draft) return;
    var ok = await mp.whenLoaded(timeoutMs || 7000);
    if (!ok) return;
    await new Promise(function (r) { setTimeout(r, 450); });
    addToLibrary(Object.assign({ thumb: mp.snapshot ? mp.snapshot(360) : null }, draft));
  }

  // Fetch the result GLB with auth, then point the live orbit viewer at the blob.
  // The viewer shows its placeholder until this resolves, so success renders
  // instantly and the real model streams in a moment later.
  async function loadPreview() {
    if (!state.previewPath) return;
    try {
      var url = await apiFileURL(state.previewPath);
      state.previewSrc = url;
      if (state.phase !== "success") return; // user navigated away meanwhile
      var mp = appEl.querySelector("[data-mp]");
      if (!mp) return;
      mp.setAttribute("src", url);
      captureToLibrary(mp, state.libDraft);
    } catch (e) { /* keep the placeholder torus if the GLB can't be fetched */ }
  }

  // =========================================================================
  // DEMO simulation (off-GPU preview of the working/success states)
  // =========================================================================
  function simGenerate() {
    stopPolling(); stopSim();
    state.jobId = "DEMO"; state.log = []; state.prog = 0; state.stageIdx = 0; state.errMsg = ""; state.errHint = "";
    state.phase = "running"; pushLog("> job DEMO accepted · tunnel secure"); render();
    var t0 = Date.now(); var lastStage = -1; var TOTAL = 12;
    var stageOrder = ["queued", "preprocessing", "generating", "making_printable"];
    state._sim = setInterval(function () {
      var e = (Date.now() - t0) / 1000;
      var idx = e < 1 ? 0 : e < 3 ? 1 : e < 8 ? 2 : 3;
      if (idx !== lastStage) { lastStage = idx; state.stageIdx = idx; pushLog(stageLog(stageOrder[idx])); }
      state.prog = Math.min(99, e / TOTAL * 100);
      patchRunning();
      if (e >= TOTAL) {
        stopSim();
        state.previewSrc = null;
        state.infoRows = [["Vertices", "40,000"], ["Faces", "80,000"], ["Size", "82 × 64 × 91 mm"], ["Watertight", state.skipPrep ? "NO (prep skipped)" : "YES"], ["Files", "STL · GLB"]];
        state.diagRows = state.skipPrep
          ? [["△", "warn", "Print-prep was skipped — mesh is not watertight."], ["△", "warn", "Overhangs up to 38° — supports recommended."]]
          : [["✓", "ok", "Watertight — 3 small holes were sealed automatically."], ["✓", "ok", "Minimum wall thickness 1.4 mm — printable on FDM and resin."], ["△", "warn", "Overhangs up to 38° on the underside — add supports."]];
        state.dlSTL = null; state.dlGLB = null;
        state.phase = "success"; render();
        // Snapshot the demo model into the Library so the gallery is demoable.
        captureToLibrary(appEl.querySelector("[data-mp]"), {
          id: "demo-" + Date.now(), jobId: "DEMO", name: baseName(state.photoName) || "Demo specimen",
          date: new Date().toISOString(), faces: 80000, vertices: 40000, watertight: !state.skipPrep,
        }, 3000);
      }
    }, 120);
  }
  function demoSTL() { return "solid solidify_demo\nendsolid solidify_demo\n"; }

  // =========================================================================
  // Event delegation
  // =========================================================================
  var ACTIONS = {
    gotoHome: function () { go("home"); },
    gotoAbout: function () { go("about"); },
    gotoStudio: function () { go("studio"); },
    logout: logout,
    browse: function () { var fi = appEl.querySelector("#file-input"); if (fi) fi.click(); },
    clearPhoto: clearPhoto,
    toggleAdv: function () { state.advOpen = !state.advOpen; render(); },
    gotoLibrary: function () { go("library"); },
    generate: generate,
    retry: retry,
    startOver: startOver,
    cancelQueue: cancelQueue,
    downloadSTL: function (e, el) { downloadResult("STL", el); },
    downloadGLB: function (e, el) { downloadResult("GLB", el); },
    dl: function (e, el) { downloadFile(el.dataset.path, el.dataset.name, el); },
    libRemove: function (e, el) { removeFromLibrary(el.dataset.id); render(); },
  };

  appEl.addEventListener("click", function (e) {
    var el = e.target.closest("[data-act]");
    if (!el) return;
    var act = el.getAttribute("data-act");
    if (act === "login") return; // handled on submit
    if (el.tagName === "BUTTON") e.preventDefault();
    if (ACTIONS[act]) ACTIONS[act](e, el);
  });

  appEl.addEventListener("submit", function (e) {
    var f = e.target.closest('[data-act="login"]');
    if (!f) return;
    e.preventDefault();
    doLogin();
  });

  appEl.addEventListener("input", function (e) {
    var el = e.target.closest("[data-model]");
    if (!el) return;
    var key = el.getAttribute("data-model");
    if (key === "detail") {
      state.detail = parseInt(el.value, 10);
      var lbl = appEl.querySelector("#detail-label"); if (lbl) lbl.textContent = num(state.detail);
    } else if (key === "skipPrep") {
      state.skipPrep = el.checked;
    } else {
      state[key] = el.value; // authName / authKey / seed / quality — no re-render (preserve focus/caret)
    }
  });
  appEl.addEventListener("change", function (e) {
    if (e.target.id === "file-input") { acceptFile(e.target.files && e.target.files[0]); e.target.value = ""; }
  });

  // Drag & drop (events bubble to #app)
  appEl.addEventListener("dragover", function (e) {
    if (!e.target.closest("[data-drop]")) return;
    e.preventDefault(); if (!state.dragOver) { state.dragOver = true; render(); }
  });
  appEl.addEventListener("dragleave", function (e) {
    if (!e.target.closest("[data-drop]")) return;
    if (state.dragOver) { state.dragOver = false; render(); }
  });
  appEl.addEventListener("drop", function (e) {
    if (!e.target.closest("[data-drop]")) return;
    e.preventDefault(); state.dragOver = false;
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    render(); acceptFile(f);
  });

  // =========================================================================
  // Polish: Lenis smooth scroll + SpotlightCard glow + Magnet CTAs
  // =========================================================================
  function initScroll() {
    if (window.gsap && window.ScrollTrigger) {
      try { window.gsap.registerPlugin(window.ScrollTrigger); } catch (e) {}
    }
    if (LIGHT || !window.Lenis) return; // mobile / reduced-motion: native scroll
    try {
      var lenis = new window.Lenis({ duration: 1.1, smoothWheel: true, wheelMultiplier: 1, touchMultiplier: 1.4 });
      window.__lenis = lenis;
      if (window.gsap && window.ScrollTrigger) {
        lenis.on("scroll", window.ScrollTrigger.update);
        window.gsap.ticker.add(function (t) { lenis.raf(t * 1000); });
        window.gsap.ticker.lagSmoothing(0);
      } else {
        var raf = function (time) { lenis.raf(time); requestAnimationFrame(raf); };
        requestAnimationFrame(raf);
      }
    } catch (e) { /* native scroll is fine */ }
  }

  // Pointer-follow spotlight on [data-spot]; subtle magnetic pull on [data-magnet].
  appEl.addEventListener("pointermove", function (e) {
    var sp = e.target.closest("[data-spot]");
    if (sp) {
      var r = sp.getBoundingClientRect();
      sp.style.setProperty("--mx", (e.clientX - r.left) + "px");
      sp.style.setProperty("--my", (e.clientY - r.top) + "px");
      sp.style.setProperty("--spot", "1");
    }
    if (!LIGHT) {
      var mg = e.target.closest("[data-magnet]");
      if (mg) {
        var mr = mg.getBoundingClientRect();
        var dx = e.clientX - (mr.left + mr.width / 2);
        var dy = e.clientY - (mr.top + mr.height / 2);
        mg.style.transition = "transform .12s ease-out";
        mg.style.transform = "translate(" + (dx * 0.16).toFixed(1) + "px," + (dy * 0.28).toFixed(1) + "px)";
      }
    }
  });
  appEl.addEventListener("pointerout", function (e) {
    var sp = e.target.closest("[data-spot]"); if (sp) sp.style.setProperty("--spot", "0");
    var mg = e.target.closest("[data-magnet]"); if (mg) { mg.style.transition = "transform .35s ease"; mg.style.transform = ""; }
  });

  // =========================================================================
  // Boot
  // =========================================================================
  async function boot() {
    initScroll();
    if (!DEMO) {
      // Load config early. If the API answers (no-auth mode, or already authed),
      // we can safely restore an in-flight job. A 401 means auth is on and we're
      // not signed in yet — just show the login screen.
      var ready = await loadConfig();
      if (ready) startHealthPoll(); // API reachable (auth off, or none) → keep queue live
      var saved = ready && localStorage.getItem(LS_KEY);
      if (saved) {
        // Restore an in-flight job: jump straight into the studio and resume polling.
        state.screen = "studio"; state.phase = "running"; state.jobId = shortId(saved);
        render(); startPolling(saved);
        return;
      }
    }
    render();
  }
  boot();
})();
