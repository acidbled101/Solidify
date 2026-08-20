"""v0 print-prep: run the 14 July 2026 make_printable.py, unmodified.

That script predates the split into trellis_core/ -- it is one self-contained
CLI with no server entry point, and its repair path differs from every later
pipeline in three ways that matter:

  * auto_orient()  hull-normal resting-pose search (base contact area, CoG
                   height, overhang area, Delaunay stability). Dropped on
                   28 Jul; no other pipeline reorients at all.
  * voxelize()     rather than voxelize_subdivide(max_iter=None)
  * defaults       voxel pitch diagonal/256 and 200,000 target faces, against
                   /1024 and 1,000,000 later -- far coarser, and the reason a
                   v0 result is ~10x smaller than a v3 one.

The file is kept BYTE-IDENTICAL to commit 69378a5, so it cannot be imported
the normal way: it lives outside the package and does `import mesh_io` as a
top-level module. This shim loads it from its own directory with that
directory on sys.path (so the 14 July mesh_io.py resolves, not
trellis_core.mesh_io), then drives its main() through argv the way the CLI
would. Nothing in either July file is edited or monkey-patched.

KNOWN LIMITATION, do not "fix" it here: v0 re-simplifies to the face budget
AFTER repairing, and never re-validates. Decimation breaks watertightness, so
the exported mesh is usually NOT watertight even though the script prints
"watertight=True" mid-run -- measured on a real mesh: 305,888 faces watertight,
then 200,000 faces not. v1 was written to address exactly this. v0 is offered
for comparison against the later pipelines, not as the safe default.
"""

import importlib.util
import io
import os
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

# The unmodified 14 July tree: make_printable.py + the mesh_io.py it imports.
JULY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "printprep_20260714",
)

_module = None
_lock = threading.Lock()


def available():
    """True if the 14 July sources are present."""
    return (os.path.exists(os.path.join(JULY_DIR, "make_printable.py"))
            and os.path.exists(os.path.join(JULY_DIR, "mesh_io.py")))


def _load():
    """Import the July script once, under a private name."""
    global _module
    if _module is not None:
        return _module
    path = os.path.join(JULY_DIR, "make_printable.py")
    if not available():
        raise ImportError(f"14 July print-prep sources not found in {JULY_DIR}")

    spec = importlib.util.spec_from_file_location("_printprep_july14", path)
    mod = importlib.util.module_from_spec(spec)
    # Its `import mesh_io` must find the July copy. Prepend rather than append:
    # a stale top-level mesh_io anywhere else on the path would otherwise win.
    sys.path.insert(0, JULY_DIR)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(JULY_DIR)
    _module = mod
    return mod


class _Tee(io.StringIO):
    """Collect the script's stdout while streaming each finished line onward.

    The July script reports progress only by printing, so this is the only way
    to narrate it in the web UI without editing it.
    """

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line and self._sink is not None:
                self._sink.append(line)
        return super().write(s)


@dataclass
class PrintableV0Result:
    """Mirrors PrintableResult's server-facing fields ("single" mode only)."""
    glb_path: str
    stl_path: str
    watertight: Optional[bool] = None
    diagnostics: Optional[dict] = None
    fidelity: Optional[dict] = None          # v0 computes none; always None
    notes: list = field(default_factory=list)
    files: list = field(default_factory=list)
    stdout: str = ""


def _parse_diagnostics(text, overhang_angle):
    """Recover the numbers the July script prints but never returns."""
    out = {"overhang_angle_deg": overhang_angle}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Overhang area"):
            try:
                out["overhang_pct"] = float(line.split(":")[1].strip().split("%")[0])
            except (IndexError, ValueError):
                pass
        elif line.startswith("Thin-wall warnings"):
            try:
                head, count = line.split(":")
                out["thin_wall_warnings"] = int(count.strip())
                out["thin_wall_threshold_mm"] = float(
                    head.split("<")[1].split("mm")[0].strip())
                out["rays_sampled"] = int(head.split("sampled")[1].split("rays")[0].strip())
            except (IndexError, ValueError):
                pass
    return out or None


def run_make_printable_v0(
    glb_path,
    *,
    output_prefix,
    target_faces: Optional[int] = None,
    overhang_angle: float = 45.0,
    solid_infill: bool = True,
    voxel_pitch: Optional[float] = None,
    notes: Optional[list] = None,
) -> PrintableV0Result:
    """Drive the July CLI's main() and adapt its output to the server's shape.

    target_faces and voxel_pitch default to None, which leaves the July
    script's OWN defaults (200,000 and diagonal/256) in force -- passing the
    server's 1,000,000/1024 would erase the difference this option exists to
    show. Set them explicitly for a like-for-like run against v1/v2/v3.
    """
    mod = _load()

    argv = ["make_printable.py", str(glb_path), "--output", str(output_prefix),
            "--overhang-angle", str(overhang_angle)]
    if solid_infill:
        argv.append("--solid-infill")          # opt-in in July, on by default now
    if target_faces is not None:
        argv += ["--target-faces", str(int(target_faces))]
    if voxel_pitch is not None:
        argv += ["--voxel-pitch", str(float(voxel_pitch))]

    collected = [] if notes is None else notes
    tee = _Tee(collected)

    # main() reads sys.argv and prints; both are process-global, so serialise.
    # The worker is single-threaded today, but a warm-up or a second worker
    # would otherwise interleave two runs' argv.
    with _lock:
        saved_argv = sys.argv
        sys.argv = argv
        try:
            with redirect_stdout(tee):
                mod.main()
        finally:
            sys.argv = saved_argv

    text = tee.getvalue()
    glb_out, stl_out = f"{output_prefix}.glb", f"{output_prefix}.stl"

    # v0 prints watertightness BEFORE decimation, so trust the written file.
    watertight = None
    if os.path.exists(glb_out):
        try:
            m = trimesh.load(glb_out, force="mesh", process=False)
            watertight = bool(m.is_watertight)
            if not watertight:
                msg = ("This mesh did not come out watertight -- a slicer may "
                       "reject it. Re-run this image with pipeline v3 for a "
                       "sealed result.")
                collected.append(msg)
        except Exception:  # noqa: BLE001
            pass

    files = [{"kind": k, "path": p} for k, p in
             (("printable_glb", glb_out), ("printable_stl", stl_out))
             if os.path.exists(p)]

    return PrintableV0Result(
        glb_path=glb_out, stl_path=stl_out, watertight=watertight,
        diagnostics=_parse_diagnostics(text, overhang_angle),
        fidelity=None, notes=list(collected), files=files, stdout=text,
    )
