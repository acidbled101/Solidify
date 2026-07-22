/* fx.js — sci-fi effect web components adapted from react-bits (LetterGlitch, DecryptedText) + Vanta NET wrapper.
   Ported verbatim from the "Solidify v2" Claude Design prototype.

   On small screens (<=760px) or when the user prefers reduced motion, the heaviest
   effects (Vanta network, constant glitch loops) downgrade to a calm static state. */
(function () {
  var LIGHT_FX = (function () {
    try {
      return window.matchMedia('(max-width: 760px)').matches ||
             window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) { return false; }
  })();
  window.__SOLIDIFY_LIGHT_FX = LIGHT_FX;

  /* <glitch-text text="..." speed="34" delay="0"> — sequential decrypt reveal on view (DecryptedText, sequential/start) */
  if (!customElements.get('glitch-text')) {
    class GlitchText extends HTMLElement {
      connectedCallback() {
        if (this._init) return; this._init = true;
        var text = this.getAttribute('text') || this.textContent || '';
        var speed = parseInt(this.getAttribute('speed') || '34', 10);
        var delay = parseInt(this.getAttribute('delay') || '0', 10);
        var chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*<>/[]{}=+';
        this.style.whiteSpace = 'pre-wrap';
        this.textContent = text;
        // Reduced-motion / mobile: just show the final text, skip the scramble.
        if (LIGHT_FX) { this.textContent = text; return; }
        var self = this;
        function run() {
          var revealed = 0;
          self._iv = setInterval(function () {
            revealed++;
            var out = '';
            for (var i = 0; i < text.length; i++) {
              out += (i < revealed || text[i] === ' ') ? text[i] : chars[Math.floor(Math.random() * chars.length)];
            }
            self.textContent = out;
            if (revealed >= text.length) { clearInterval(self._iv); self.textContent = text; }
          }, speed);
        }
        var io = new IntersectionObserver(function (es) {
          es.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); setTimeout(run, delay); } });
        }, { threshold: 0.1 });
        io.observe(this);
      }
      disconnectedCallback() { clearInterval(this._iv); }
    }
    customElements.define('glitch-text', GlitchText);
  }

  /* <letter-glitch colors="#0b2f2b,#177f74,#35f2e2" speed="50"> — glitching letter grid canvas (LetterGlitch port) */
  if (!customElements.get('letter-glitch')) {
    var CHARS = Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$&*()-_+=/[]{};:<>.,0123456789');
    function hexToRgb(hex) {
      var r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return r ? { r: parseInt(r[1], 16), g: parseInt(r[2], 16), b: parseInt(r[3], 16) } : null;
    }
    class LetterGlitch extends HTMLElement {
      connectedCallback() {
        if (this._init) return; this._init = true;
        var colors = (this.getAttribute('colors') || '#0b2f2b,#177f74,#35f2e2').split(',');
        var glitchSpeed = parseInt(this.getAttribute('speed') || '50', 10);
        this.style.cssText += ';position:absolute;inset:0;overflow:hidden;display:block;pointer-events:none';
        var canvas = document.createElement('canvas');
        canvas.style.cssText = 'display:block;width:100%;height:100%';
        this.appendChild(canvas);
        if (this.getAttribute('vignette') !== 'false') {
          var v = document.createElement('div');
          v.style.cssText = 'position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle, rgba(2,13,12,0) 40%, rgba(2,13,12,1) 100%)';
          this.appendChild(v);
        }
        var ctx = canvas.getContext('2d');
        var charWidth = 10, charHeight = 20, fontSize = 16;
        var letters = [], cols = 0, rows = 0, last = Date.now(), self = this;
        function rc() { return CHARS[Math.floor(Math.random() * CHARS.length)]; }
        function rcol() { return colors[Math.floor(Math.random() * colors.length)]; }
        function resize() {
          var dpr = window.devicePixelRatio || 1;
          var rect = self.getBoundingClientRect();
          canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          cols = Math.ceil(rect.width / charWidth); rows = Math.ceil(rect.height / charHeight);
          letters = Array.from({ length: cols * rows }, function () {
            return { char: rc(), color: rcol(), target: rcol(), p: 1 };
          });
        }
        function draw() {
          var rect = self.getBoundingClientRect();
          ctx.clearRect(0, 0, rect.width, rect.height);
          ctx.font = fontSize + 'px monospace'; ctx.textBaseline = 'top';
          for (var i = 0; i < letters.length; i++) {
            ctx.fillStyle = letters[i].color;
            ctx.fillText(letters[i].char, (i % cols) * charWidth, Math.floor(i / cols) * charHeight);
          }
        }
        function tick() {
          if (!self.isConnected) return;
          var now = Date.now();
          if (now - last >= glitchSpeed) {
            var n = Math.max(1, Math.floor(letters.length * 0.05));
            for (var k = 0; k < n; k++) {
              var i = Math.floor(Math.random() * letters.length);
              if (!letters[i]) continue;
              letters[i].char = rc(); letters[i].target = rcol(); letters[i].p = 0;
            }
            last = now;
          }
          var redraw = false;
          letters.forEach(function (l) {
            if (l.p < 1) {
              l.p = Math.min(1, l.p + 0.05);
              var a = hexToRgb(l.color.startsWith('#') ? l.color : l.target), b = hexToRgb(l.target);
              if (a && b) l.color = 'rgb(' + Math.round(a.r + (b.r - a.r) * l.p) + ',' + Math.round(a.g + (b.g - a.g) * l.p) + ',' + Math.round(a.b + (b.b - a.b) * l.p) + ')';
              else l.color = l.target;
              redraw = true;
            }
          });
          if (redraw || now === last) draw();
          self._raf = requestAnimationFrame(tick);
        }
        this._ro = new ResizeObserver(function () { requestAnimationFrame(function () { resize(); draw(); }); });
        this._ro.observe(this);
        resize(); draw();
        if (LIGHT_FX) return; // draw one static frame, skip the animation loop
        tick();
      }
      disconnectedCallback() { cancelAnimationFrame(this._raf); if (this._ro) this._ro.disconnect(); }
    }
    customElements.define('letter-glitch', LetterGlitch);
  }

  /* <vanta-bg> — Vanta NET animated 3D network background (waits for global THREE + VANTA) */
  if (!customElements.get('vanta-bg')) {
    class VantaBg extends HTMLElement {
      connectedCallback() {
        this.style.cssText += ';position:absolute;inset:0;display:block;overflow:hidden';
        // Mobile / reduced motion: skip the heavy WebGL network, keep a calm gradient backdrop.
        if (LIGHT_FX) {
          this.style.background = 'radial-gradient(circle at 30% 20%, rgba(53,242,226,.10), transparent 60%), #010807';
          return;
        }
        var self = this;
        var color = parseInt((this.getAttribute('color') || '35f2e2').replace('#', ''), 16);
        var bg = parseInt((this.getAttribute('background') || '020d0c').replace('#', ''), 16);
        this._poll = setInterval(function () {
          if (window.VANTA && window.VANTA.NET && window.THREE && self.isConnected) {
            clearInterval(self._poll); self._poll = null;
            try {
              self._fx = window.VANTA.NET({
                el: self, mouseControls: true, touchControls: true, gyroControls: false,
                minHeight: 200, minWidth: 200, scale: 1, scaleMobile: 1,
                color: color, backgroundColor: bg,
                points: 11, maxDistance: 23, spacing: 17, showDots: true
              });
            } catch (e) { /* background stays plain */ }
          }
        }, 150);
        setTimeout(function () { if (self._poll) clearInterval(self._poll); }, 20000);
      }
      disconnectedCallback() {
        if (this._poll) clearInterval(this._poll);
        if (this._fx) { try { this._fx.destroy(); } catch (e) {} this._fx = null; }
      }
    }
    customElements.define('vanta-bg', VantaBg);
  }

  /* <specimen-scan photo="..." mesh="..."> — duotone specimen; a sweeping scan line converts 2D photo → 3D mesh as it passes; interconnected circuit-trace overlay */
  if (!customElements.get('specimen-scan')) {
    class SpecimenScan extends HTMLElement {
      connectedCallback() {
        if (this._init) return; this._init = true;
        var photo = this.getAttribute('photo') || '';
        var mesh = this.getAttribute('mesh') || '';
        this.style.cssText += ';position:relative;display:block;overflow:hidden;background:#0FD8C7';
        if (!document.getElementById('specimen-scan-kf')) {
          var st = document.createElement('style'); st.id = 'specimen-scan-kf';
          st.textContent = '@keyframes ssLine{0%,14%{top:2%}50%,64%{top:96%}100%{top:2%}}' +
            '@keyframes ssReveal{0%,14%{clip-path:inset(0 0 98% 0)}50%,64%{clip-path:inset(0 0 4% 0)}100%{clip-path:inset(0 0 98% 0)}}' +
            '@keyframes ssFlick{0%,88%,100%{opacity:.95}90%{opacity:.3}94%{opacity:.75}97%{opacity:.95}}';
          document.head.appendChild(st);
        }
        var sq = function (x, y) { return '<rect x="' + x + '" y="' + y + '" width="3" height="3" fill="rgba(255,255,255,.95)"/>'; };
        var R = 'fill="none" stroke="rgba(255,255,255,.9)" stroke-width="1.2" vector-effect="non-scaling-stroke"';
        var L = 'fill="none" stroke="rgba(255,255,255,.75)" stroke-width="1" vector-effect="non-scaling-stroke"';
        var host = document.createElement('div');
        host.style.cssText = 'position:absolute;inset:0;background:#0FD8C7';
        host.innerHTML =
          '<img src="' + photo + '" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;filter:grayscale(1) contrast(1.25) brightness(1.05);mix-blend-mode:multiply;padding:8%"/>' +
          '<div style="position:absolute;inset:0;background-image:radial-gradient(rgba(2,32,28,.16) 1px, transparent 1.5px);background-size:8px 8px"></div>' +
          '<img src="' + mesh + '" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#0FD8C7;filter:grayscale(1) sepia(1) hue-rotate(125deg) saturate(2.4) brightness(.95) contrast(1.1);clip-path:inset(0 0 98% 0);animation:ssReveal 8s ease-in-out infinite"/>' +
          '<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;filter:drop-shadow(0 0 3px rgba(255,255,255,.8));animation:ssFlick 6s linear infinite">' +
            /* boxes tracing the subject silhouette, sharing edges */
            '<rect x="26" y="36" width="26" height="18" ' + R + '/>' +
            '<rect x="50" y="34" width="18" height="14" ' + R + '/>' +
            '<rect x="64" y="40" width="16" height="14" ' + R + '/>' +
            '<rect x="18" y="44" width="16" height="14" ' + R + '/>' +
            '<rect x="34" y="50" width="24" height="16" ' + R + '/>' +
            '<rect x="58" y="52" width="16" height="12" ' + R + '/>' +
            '<rect x="26" y="62" width="20" height="10" ' + R + '/>' +
            '<rect x="46" y="64" width="16" height="8" ' + R + '/>' +
            '<rect x="70" y="32" width="10" height="8" ' + R + '/>' +
            '<rect x="12" y="52" width="10" height="10" ' + R + '/>' +
            /* nested inner boxes */
            '<rect x="30" y="40" width="10" height="8" ' + L + '/>' +
            '<rect x="54" y="38" width="8" height="6" ' + L + '/>' +
            '<rect x="68" y="44" width="8" height="6" ' + L + '/>' +
            '<rect x="22" y="48" width="8" height="6" ' + L + '/>' +
            '<rect x="38" y="54" width="10" height="8" ' + L + '/>' +
            '<rect x="60" y="56" width="8" height="6" ' + L + '/>' +
            /* bitmap stair-steps */
            '<rect x="52" y="30" width="4" height="4" ' + L + '/>' +
            '<rect x="56" y="26" width="4" height="4" ' + L + '/>' +
            '<rect x="60" y="22" width="4" height="4" ' + L + '/>' +
            '<rect x="22" y="72" width="4" height="4" ' + L + '/>' +
            '<rect x="18" y="68" width="4" height="4" ' + L + '/>' +
            '<rect x="14" y="64" width="4" height="4" ' + L + '/>' +
            /* circuit traces out to the edges */
            '<polyline points="80,44 92,44 92,30" ' + L + '/>' +
            '<polyline points="12,56 4,56 4,42" ' + L + '/>' +
            '<polyline points="50,72 50,88 64,88" ' + L + '/>' +
            '<polyline points="36,36 36,20 24,20" ' + L + '/>' +
            sq(24.5, 34.5) + sq(50.5, 32.5) + sq(62.5, 38.5) + sq(16.5, 42.5) + sq(32.5, 48.5) +
            sq(56.5, 50.5) + sq(24.5, 60.5) + sq(44.5, 62.5) + sq(78.5, 30.5) + sq(68.5, 52.5) +
            sq(10.5, 50.5) + sq(90.5, 28.5) + sq(2.5, 40.5) + sq(62.5, 86.5) + sq(22.5, 18.5) + sq(48.5, 86.5) +
          '</svg>' +
          '<div style="position:absolute;left:0;right:0;top:2%;height:2px;background:rgba(255,255,255,.95);box-shadow:0 0 20px rgba(255,255,255,1), 0 0 40px rgba(53,242,226,.8);animation:ssLine 8s ease-in-out infinite"></div>';
        this.appendChild(host);
        this._host = host;
      }
      disconnectedCallback() { this._init = false; if (this._host) { this._host.remove(); this._host = null; } }
    }
    customElements.define('specimen-scan', SpecimenScan);
  }
})();
