// viewer.js -- side-by-side three.js viewer for two candidate meshes
// (reference vs delta), with synchronized orbit cameras so rotating one
// rotates both -- makes it possible to actually compare the two shapes
// rather than wrestling with two independent cameras.
import * as THREE from "three";
import { GLTFLoader } from "/static/vendor/loaders/GLTFLoader.js";
import { OrbitControls } from "/static/vendor/loaders/OrbitControls.js";

const loader = new GLTFLoader();

class Pane {
  constructor(canvas, label, color) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1116);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0.9, 0.7, 0.9);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    // three.js r155+ switched to physically-correct light units, where the
    // old "intensity: 1-2" convention this was first written against reads
    // as near-dark -- MeshStandardMaterial under PBR lighting needs
    // intensities an order of magnitude higher for a flat matte material to
    // read as lit rather than a silhouette. No tone mapping (default) so
    // these numbers map roughly linearly to brightness.
    this.renderer.toneMapping = THREE.NoToneMapping;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 3.5);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 4.5);
    key.position.set(2, 3, 2);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 2.5);
    fill.position.set(-2, 1, -1.5);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(color, 1.5);
    rim.position.set(-1, -2, -2);
    this.scene.add(rim);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.2));

    const grid = new THREE.GridHelper(1.2, 12, 0x2a3540, 0x1a222a);
    this.scene.add(grid);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.meshGroup = null;
    this.label = label;
    this.linked = null; // another Pane whose camera mirrors this one
  }

  linkTo(other) {
    this.linked = other;
    this.controls.addEventListener("change", () => {
      if (!this.linked || this.linked._syncing) return;
      this._syncing = true;
      this.linked.camera.position.copy(this.camera.position);
      this.linked.camera.quaternion.copy(this.camera.quaternion);
      this.linked.controls.target.copy(this.controls.target);
      this.linked.controls.update();
      this._syncing = false;
    });
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clearMesh() {
    if (this.meshGroup) {
      this.scene.remove(this.meshGroup);
      this.meshGroup.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      });
      this.meshGroup = null;
    }
  }

  async loadGlb(url, color) {
    this.clearMesh();
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => {
          const group = gltf.scene;
          group.traverse((o) => {
            if (o.isMesh) {
              // devlab/trace.py exports candidate meshes via trimesh's
              // default GLB export, which writes POSITION only -- no
              // NORMAL attribute at all (verified directly against an
              // exported file). Without normals PBR lighting has nothing
              // to shade against and the mesh renders as a flat black
              // silhouette regardless of light intensity, which is
              // genuinely what was happening here (traced past several
              // failed intensity-tuning attempts before finding this).
              if (!o.geometry.attributes.normal) {
                o.geometry.computeVertexNormals();
              }
              o.material = new THREE.MeshStandardMaterial({
                color, metalness: 0.0, roughness: 0.75, flatShading: false,
              });
            }
          });
          // center + normalize scale so ref/delta (near-identical, tiny
          // perturbations) sit at comparable framing regardless of extents.
          const box = new THREE.Box3().setFromObject(group);
          const size = new THREE.Vector3();
          box.getSize(size);
          const center = new THREE.Vector3();
          box.getCenter(center);
          group.position.sub(center);
          const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
          const scale = 0.85 / maxDim;
          group.scale.setScalar(scale);

          this.meshGroup = group;
          this.scene.add(group);
          resolve(gltf);
        },
        undefined,
        (err) => reject(err),
      );
    });
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

export class DualMeshViewer {
  constructor(canvasLeft, canvasRight) {
    this.left = new Pane(canvasLeft, "reference", 0x3ed6c4);
    this.right = new Pane(canvasRight, "delta", 0xff9f45);
    this.left.linkTo(this.right);
    this.right.linkTo(this.left);

    this._running = true;
    const loop = () => {
      if (!this._running) return;
      this.left.resize();
      this.right.resize();
      this.left.render();
      this.right.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    window.addEventListener("resize", () => {
      this.left.resize();
      this.right.resize();
    });
  }

  async loadLeft(url) {
    try {
      await this.left.loadGlb(url, 0x3ed6c4);
    } catch (e) {
      console.warn("[viewer] failed to load left mesh", url, e);
    }
  }

  async loadRight(url) {
    try {
      await this.right.loadGlb(url, 0xff9f45);
    } catch (e) {
      console.warn("[viewer] failed to load right mesh", url, e);
    }
  }

  dispose() {
    this._running = false;
    this.left.clearMesh();
    this.right.clearMesh();
  }
}
