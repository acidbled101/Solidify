/* <model-preview> — rotatable three.js viewer with a "forming" mode.
   Attributes: mode="orbit|forming", progress="0..100", autorotate="true|false", src="<glb-url>"

   Ported from the "Solidify v2" Claude Design prototype and extended:
   - With no `src`, renders the stylised torus-knot placeholder (used for the "forming"
     loading animation and design previews).
   - With a `src` GLB url in `orbit` mode, loads the real generated model and applies the
     same cyan emissive material + wireframe overlay so the result matches the aesthetic. */
(function () {
  if (customElements.get('model-preview')) return;
  var THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js';
  var GLTF_URL = 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/loaders/GLTFLoader.js';
  var STL_URL = 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/loaders/STLLoader.js';
  var threeP = null;
  function loadThree() { return threeP || (threeP = import(THREE_URL)); }

  class ModelPreview extends HTMLElement {
    static get observedAttributes() { return ['progress', 'mode', 'autorotate', 'src']; }
    constructor() {
      super();
      this._prog = 1; this._mode = 'orbit'; this._auto = true; this._src = null;
      this._rotY = 0.7; this._rotX = 0.35; this._dist = 3.9; this._dragging = false;
    }
    attributeChangedCallback(n, _o, v) {
      if (n === 'progress') this._prog = Math.max(0, Math.min(1, (parseFloat(v) || 0) / 100));
      if (n === 'mode') this._mode = v || 'orbit';
      if (n === 'autorotate') this._auto = v !== 'false';
      if (n === 'src') { this._src = v || null; if (this._THREE) this._loadModel(); }
    }
    connectedCallback() {
      if (!this.style.display) this.style.display = 'block';
      this._init();
    }
    disconnectedCallback() {
      this._dead = true;
      cancelAnimationFrame(this._raf);
      if (this._ro) this._ro.disconnect();
      if (this._io) this._io.disconnect();
      if (this._renderer) this._renderer.dispose();
    }
    async _init() {
      var THREE;
      try { THREE = await loadThree(); }
      catch (e) {
        this.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;font-family:monospace;font-size:12px;color:#57c9bf">3D preview needs a network connection</div>';
        return;
      }
      if (this._dead || this._renderer) return;
      this._THREE = THREE;
      // preserveDrawingBuffer lets snapshot() read the canvas for the Library.
      var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      this._renderer = renderer;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;touch-action:none;cursor:grab';
      this.appendChild(renderer.domElement);

      var scene = new THREE.Scene();
      var cam = new THREE.PerspectiveCamera(42, 1, 0.1, 60);
      var group = new THREE.Group(); scene.add(group);
      this._scene = scene; this._group = group;

      var geo = new THREE.TorusKnotGeometry(1, 0.32, 240, 40).toNonIndexed();
      geo.computeVertexNormals();
      var mat = this._skinMaterial();
      var mesh = new THREE.Mesh(geo, mat); group.add(mesh);

      var wgeo = new THREE.WireframeGeometry(new THREE.TorusKnotGeometry(1, 0.32, 110, 18));
      var wire = new THREE.LineSegments(wgeo, new THREE.LineBasicMaterial({ color: 0x67ffe9, transparent: true, opacity: 0.18 }));
      wire.scale.setScalar(1.004); group.add(wire);
      this._placeholder = { mesh: mesh, wire: wire };

      scene.add(new THREE.HemisphereLight(0xbffff4, 0x02100e, 1.15));
      var d1 = new THREE.DirectionalLight(0x9ffcf0, 1.7); d1.position.set(3, 4, 2); scene.add(d1);
      var d2 = new THREE.DirectionalLight(0x2affd0, 0.8); d2.position.set(-3, -2, -4); scene.add(d2);

      var total = geo.attributes.position.count;
      var wtotal = wgeo.attributes.position.count;

      var self = this, px = 0, py = 0, dom = renderer.domElement;
      dom.addEventListener('pointerdown', function (e) { self._dragging = true; px = e.clientX; py = e.clientY; dom.setPointerCapture(e.pointerId); dom.style.cursor = 'grabbing'; });
      dom.addEventListener('pointermove', function (e) {
        if (!self._dragging) return;
        self._rotY += (e.clientX - px) * 0.008;
        self._rotX = Math.max(-1.4, Math.min(1.4, self._rotX + (e.clientY - py) * 0.008));
        px = e.clientX; py = e.clientY;
      });
      function end() { self._dragging = false; dom.style.cursor = 'grab'; }
      dom.addEventListener('pointerup', end);
      dom.addEventListener('pointercancel', end);
      dom.addEventListener('wheel', function (e) { e.preventDefault(); self._dist = Math.max(2.3, Math.min(8, self._dist + e.deltaY * 0.003)); }, { passive: false });

      function size() {
        var w = self.clientWidth || 300, h = self.clientHeight || 300;
        renderer.setSize(w, h, false);
        cam.aspect = w / h; cam.updateProjectionMatrix();
      }
      this._ro = new ResizeObserver(size); this._ro.observe(this); size();

      // Pause rendering while scrolled off-screen (the animated loop otherwise
      // keeps drawing an invisible canvas — wasted GPU/battery, esp. the Home hero).
      this._onscreen = true;
      if (window.IntersectionObserver) {
        this._io = new IntersectionObserver(function (es) { self._onscreen = es[0].isIntersecting; });
        this._io.observe(this);
      }

      // If a real model was requested before three finished loading, load it now.
      if (this._src) this._loadModel();

      function loop() {
        if (self._dead) return;
        self._raf = requestAnimationFrame(loop);
        if (self._onscreen === false || document.hidden) return; // skip work off-screen / hidden tab
        if (self._auto && !self._dragging) self._rotY += 0.0045;
        group.rotation.set(self._rotX, self._rotY, 0);
        cam.position.set(0, 0.15, self._dist); cam.lookAt(0, 0, 0);
        var showReal = self._real && self._mode !== 'forming';
        self._placeholder.mesh.visible = !showReal;
        self._placeholder.wire.visible = !showReal;
        if (self._real) self._real.visible = showReal;
        if (!showReal && self._mode === 'forming') {
          var p = self._prog;
          geo.setDrawRange(0, Math.floor(total * p / 3) * 3);
          wgeo.setDrawRange(0, Math.floor(wtotal * Math.min(1, p * 1.2)));
          wire.material.opacity = 0.4;
          mat.emissiveIntensity = 0.9 + Math.sin(performance.now() * 0.004) * 0.35;
        } else if (!showReal) {
          geo.setDrawRange(0, Infinity);
          wgeo.setDrawRange(0, Infinity);
          wire.material.opacity = 0.18;
          mat.emissiveIntensity = 0.55;
        }
        renderer.render(scene, cam);
      }
      loop();
    }

    _skinMaterial() {
      var THREE = this._THREE;
      return new THREE.MeshStandardMaterial({ color: 0x10c2b0, metalness: 0.3, roughness: 0.32, emissive: 0x043d37, emissiveIntensity: 0.55 });
    }
    _addWire(mesh) {
      var THREE = this._THREE;
      // Skip the wireframe overlay on very dense meshes (too many segments to be readable / cheap).
      if (mesh.geometry.attributes.position.count > 150000) return;
      var wire = new THREE.LineSegments(
        new THREE.WireframeGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: 0x67ffe9, transparent: true, opacity: 0.12 })
      );
      mesh.add(wire);
    }
    _finalize(root) {
      var THREE = this._THREE;
      if (this._dead) return;
      if (this._real) { this._group.remove(this._real); this._real = null; }
      // Guarantee lit shading: any mesh without a NORMAL attribute renders
      // near-black under the emissive material, so compute normals here (works
      // for both the GLB and STL paths; idempotent if normals already exist).
      root.traverse(function (o) {
        if (o.isMesh && o.geometry && !o.geometry.attributes.normal) {
          o.geometry.computeVertexNormals();
          if (o.material) o.material.needsUpdate = true;
        }
      });
      // Center + scale to roughly match the placeholder framing.
      var box = new THREE.Box3().setFromObject(root);
      var sphere = box.getBoundingSphere(new THREE.Sphere());
      var s = sphere.radius > 0 ? 1.4 / sphere.radius : 1;
      root.scale.setScalar(s);
      box.setFromObject(root);
      root.position.sub(box.getCenter(new THREE.Vector3()));
      this._real = root;
      this._group.add(root);
    }
    async _loadModel() {
      var THREE = this._THREE;
      if (!THREE || !this._src || this._loadedSrc === this._src) return;
      this._loadedSrc = this._src;
      var self = this;
      var isSTL = /\.stl(\?|$)/i.test(this._src);
      try {
        if (isSTL) {
          var smod = await import(STL_URL);
          if (self._dead) return;
          new smod.STLLoader().load(self._src, function (geometry) {
            if (self._dead) return;
            geometry.computeVertexNormals();
            var mesh = new THREE.Mesh(geometry, self._skinMaterial());
            self._addWire(mesh);
            // STL is typically Z-up; three is Y-up. Stand the model upright, then
            // wrap in a group so orbit rotation composes cleanly.
            mesh.rotation.x = -Math.PI / 2;
            var g = new THREE.Group(); g.add(mesh);
            self._finalize(g);
          }, undefined, function () { /* keep placeholder */ });
        } else {
          var mod = await import(GLTF_URL);
          if (self._dead) return;
          new mod.GLTFLoader().load(self._src, function (gltf) {
            if (self._dead) return;
            var root = gltf.scene;
            root.traverse(function (o) {
              if (o.isMesh && o.geometry) { o.material = self._skinMaterial(); self._addWire(o); }
            });
            self._finalize(root); // computes normals for any mesh missing them
          }, undefined, function () { /* keep placeholder */ });
        }
      } catch (e) { /* keep placeholder */ }
    }

    // Wait until the real model has loaded (or time out), for snapshotting.
    whenLoaded(timeoutMs) {
      var self = this, t0 = Date.now(), lim = timeoutMs || 6000;
      return new Promise(function (resolve) {
        (function tick() {
          if (self._dead) return resolve(false);
          if (self._real) return resolve(true);
          if (Date.now() - t0 > lim) return resolve(false);
          setTimeout(tick, 150);
        })();
      });
    }

    // Downscaled JPEG data URL of the current view — used for Library thumbnails.
    snapshot(maxW) {
      try {
        var c = this._renderer && this._renderer.domElement;
        if (!c || !c.width) return null;
        var w = maxW || 360, h = Math.max(1, Math.round(w * (c.height / c.width)));
        var oc = document.createElement('canvas'); oc.width = w; oc.height = h;
        var ctx = oc.getContext('2d');
        ctx.fillStyle = '#02100e'; ctx.fillRect(0, 0, w, h); // opaque bg (canvas is alpha)
        ctx.drawImage(c, 0, 0, w, h);
        return oc.toDataURL('image/jpeg', 0.72);
      } catch (e) { return null; }
    }
  }
  customElements.define('model-preview', ModelPreview);
})();
