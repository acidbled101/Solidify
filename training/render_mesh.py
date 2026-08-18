"""Offscreen mesh renderer for building conditioning images.

WHY RENDER RATHER THAN SCRAPE
-----------------------------
TRELLIS.2 is image-conditioned, so every dataset entry needs a picture. The
obvious source -- the photo on the model's Thingiverse page -- is the wrong
one: those are typically photographs of a *printed* object, often with
supports still attached, on a printer bed, painted, assembled with other
parts, or showing several copies at once. The image and the mesh then do not
correspond, and the pair teaches the model a relationship that isn't there.

Rendering the mesh ourselves gives exact correspondence and puts the image in
the same domain TRELLIS was trained on (renders, not photographs).

WHY moderngl
------------
It is the only renderer installed here that works headless. pyrender, pyglet,
PyOpenGL, vtk and open3d are all absent, and trimesh's `save_image` is a
pyglet wrapper so it fails too. moderngl opens a standalone Metal-backed
OpenGL 4.1 context on Apple Silicon with no display attached, which is what an
unattended dataset build needs.

Renders RGBA with a transparent background, so the alpha channel *is* the
object mask. That skips rembg entirely -- one less failure mode in a loop that
has to run unattended over hundreds of meshes, and a cleaner matte than
background removal would produce.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

_ctx = None

_VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    v_normal = mat3(model) * in_normal;
    v_pos = (model * vec4(in_pos, 1.0)).xyz;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

# Two-light Lambertian plus a rim term and a mild vertical gradient. Flat
# shading would leave large facets reading as one tone, which gives DINO very
# little to key on; the rim term keeps silhouette edges legible against a
# transparent background.
_FRAG = """
#version 330
uniform vec3 light_dir;
uniform vec3 fill_dir;
uniform vec3 eye;
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 view = normalize(eye - v_pos);
    if (dot(n, view) < 0.0) n = -n;          // meshes with inconsistent winding
    float key  = max(dot(n, normalize(light_dir)), 0.0);
    float fill = max(dot(n, normalize(fill_dir)), 0.0);
    float rim  = pow(1.0 - max(dot(n, view), 0.0), 3.0);
    vec3 base = vec3(0.78, 0.78, 0.80);
    vec3 col = base * (0.22 + 0.68 * key + 0.22 * fill) + vec3(0.25) * rim;
    f_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""


def _context():
    global _ctx
    if _ctx is None:
        import moderngl
        _ctx = moderngl.create_standalone_context()
    return _ctx


def _look_at(eye, target, up) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def render_views(
    mesh,
    n_views: int = 1,
    size: int = 518,
    supersample: int = 2,
    elevation_deg: float = 20.0,
    azimuth_deg: float = 35.0,
    distance: Optional[float] = None,
    fov_deg: float = 40.0,
    margin: float = 1.12,
) -> List["Image.Image"]:
    """Render `n_views` RGBA images of a trimesh, evenly spaced in azimuth.

    With `distance=None` (the default) the camera is fitted to each mesh's
    bounding sphere so every object subtends the same angle regardless of its
    shape or original scale. That consistency is the point: a fixed distance
    frames a compact object tightly and a sprawling one loosely (measured 22%
    vs 11% pixel coverage on two real models), which puts apparent object size
    into the dataset as a confound the model can learn instead of geometry.

    `size` defaults to 518: DINOv3's patch grid divides it exactly, so the
    conditioning encoder does not resample and we do not throw away detail we
    just rendered.
    """
    import moderngl
    from PIL import Image

    ctx = _context()
    ss = max(1, int(supersample))
    w = h = size * ss

    verts = np.asarray(mesh.vertices, dtype="f4")
    faces = np.asarray(mesh.faces, dtype="i4")

    # Centre and scale into the unit cube.
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    verts = (verts - (lo + hi) / 2) / max((hi - lo).max(), 1e-9)

    # Fit the camera to the bounding sphere: half-angle asin(r/d) must be
    # within the half-fov, so d = r / sin(fov/2), times a margin so the
    # silhouette never touches the frame edge.
    if distance is None:
        radius = float(np.linalg.norm(verts, axis=1).max())
        distance = margin * radius / math.sin(math.radians(fov_deg) / 2)

    # Per-vertex normals by area-weighted face accumulation. trimesh can supply
    # these, but it recomputes adjacency on meshes that may be non-manifold and
    # is markedly slower across hundreds of models.
    fv = verts[faces]
    fn = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    lens = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.maximum(lens, 1e-9)

    prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
    vbo = ctx.buffer(np.hstack([verts, vn]).astype("f4").tobytes())
    ibo = ctx.buffer(faces.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_normal")], ibo)

    colour = ctx.renderbuffer((w, h), components=4)
    depth = ctx.depth_renderbuffer((w, h))
    fbo = ctx.framebuffer(color_attachments=[colour], depth_attachment=depth)
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    model = np.eye(4, dtype="f4")
    proj = _perspective(fov_deg, 1.0, 0.05, 20.0)
    out = []
    try:
        for i in range(n_views):
            az = math.radians(azimuth_deg + i * 360.0 / n_views)
            el = math.radians(elevation_deg)
            eye = np.array([
                distance * math.cos(el) * math.sin(az),
                distance * math.sin(el),
                distance * math.cos(el) * math.cos(az),
            ], dtype="f4")
            view = _look_at(eye, np.zeros(3, dtype="f4"), np.array([0, 1, 0], dtype="f4"))
            prog["mvp"].write((proj @ view @ model).T.astype("f4").tobytes())
            prog["model"].write(model.T.astype("f4").tobytes())
            prog["light_dir"].value = tuple((eye / np.linalg.norm(eye) + np.array([0.3, 0.8, 0.2])).tolist())
            prog["fill_dir"].value = (-0.5, 0.2, -0.7)
            prog["eye"].value = tuple(eye.tolist())

            # Transparent clear: alpha becomes the object mask, so no rembg.
            ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            vao.render()

            img = Image.frombytes("RGBA", (w, h), fbo.read(components=4, dtype="f1"))
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if ss > 1:
                img = img.resize((size, size), Image.LANCZOS)
            out.append(img)
    finally:
        for obj in (vao, vbo, ibo, fbo, colour, depth):
            try:
                obj.release()
            except Exception:
                pass
    return out


def coverage(img) -> float:
    """Fraction of pixels covered by the object.

    The build loop uses this as a sanity gate: a render that covers almost
    nothing means the camera framing failed (degenerate bounds, a stray vertex
    at infinity), and such an entry would poison a training pair silently.
    """
    a = np.asarray(img)[..., 3]
    return float((a > 8).mean())
