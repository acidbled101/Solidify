"""CPU build of o-voxel's mesh -> flexible-dual-grid voxelizer, for Apple Silicon.

WHY THIS EXISTS
---------------
`o_voxel.convert.mesh_to_flexible_dual_grid` is the inverse of the shape
decoder: it turns a triangle mesh into the (dual_vertices, intersected)
representation that `FlexiDualGridVaeEncoder` consumes. That is the only way to
get an arbitrary mesh -- a downloaded printable model, say -- into TRELLIS.2's
SLat space.

The `o_voxel` python package ships in the venv, but its native `_C` extension
does not build here: `o-voxel/setup.py` declares a `CUDAExtension` over a
source list containing `.cu` files, so it needs nvcc. Importing anything from
`o_voxel.convert` then fails with `NameError: name '_C' is not defined`.

The function we need is in `src/convert/flexible_dual_grid.cpp`, which contains
**zero CUDA references** -- plain C++17 with Eigen and STL. This module JIT-
compiles that single translation unit into a CPU-only extension, applying three
portability fixes that nvcc/gcc tolerate and clang does not:

  1. `-Wno-c++11-narrowing` -- the source brace-initializes `int4` from
     `size_t` neighbour indices. Narrowing in a braced init list is ill-formed
     by the standard; gcc/nvcc warn, clang errors. The values are voxel indices
     well inside int range, so downgrading is safe.
  2. `d`-suffixed floating literals (`0.0d`, `1e-6d`) are a gcc/nvcc extension
     and invalid C++. Rewritten to unsuffixed doubles. The substitution is
     deliberately anchored to require a decimal point or an exponent -- a naive
     `\\d+d` pattern also rewrites `Eigen::Vector3d` to `Eigen::Vector3` and
     produces a wall of template errors.
  3. `src/convert` must be on the include path; the file includes "api.h"
     relative to its own directory.

Eigen is a git submodule of o-voxel that is not initialised in this checkout,
so `eigen_include` must point at headers from elsewhere. Header-only: any 3.4.x
source tree works.

Nothing here is CUDA-specific in reverse either -- the same build works on a
Linux CPU. It is simply the subset of o-voxel that does not need a GPU.
"""

import os
import re
import shutil
from typing import Optional, Tuple

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OVOX_SRC = os.path.join(_REPO, "TRELLIS.2", "o-voxel", "src")

# Anchored so `Vector3d`, `int4`, and friends are left alone: the literal must
# carry a '.' or an exponent to be rewritten.
_D_SUFFIX = re.compile(r"(\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)d\b")

_module = None


def _patch_source(dst_dir: str) -> str:
    src = os.path.join(_OVOX_SRC, "convert", "flexible_dual_grid.cpp")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"o-voxel source not found at {src}. This module needs the vendored "
            "TRELLIS.2/o-voxel checkout, not just the installed o_voxel wheel."
        )
    with open(src) as f:
        text = f.read()
    patched, n = _D_SUFFIX.subn(r"\1", text)
    dst = os.path.join(dst_dir, "flexible_dual_grid_cpu.cpp")
    with open(dst, "w") as f:
        f.write(patched)

    binding = os.path.join(dst_dir, "ext_cpu.cpp")
    with open(binding, "w") as f:
        f.write(
            '#include <torch/extension.h>\n'
            '#include "convert/api.h"\n\n'
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
            '    m.def("mesh_to_flexible_dual_grid_cpu", &mesh_to_flexible_dual_grid_cpu,\n'
            "          py::call_guard<py::gil_scoped_release>());\n"
            "}\n"
        )
    return dst


def _build_with_setuptools(build_dir: str, eigen: str, verbose: bool):
    """Compile via setuptools rather than torch's `cpp_extension.load`.

    `load()` requires ninja, which is not a dependency of this project and is
    not installed in the venv. setuptools' build_ext needs only the compiler
    that is already there, so this keeps the module dependency-free.
    Cached: an existing .so is imported directly on subsequent calls.
    """
    import glob
    import importlib.util
    import subprocess

    existing = glob.glob(os.path.join(build_dir, "ovox_cpu*.so"))
    if not existing:
        setup_py = os.path.join(build_dir, "setup.py")
        with open(setup_py, "w") as f:
            f.write(
                "from setuptools import setup\n"
                "from torch.utils.cpp_extension import CppExtension, BuildExtension\n"
                "setup(name='ovox_cpu', ext_modules=[CppExtension(\n"
                "    name='ovox_cpu',\n"
                f"    sources={[os.path.join(build_dir, 'ext_cpu.cpp'), os.path.join(build_dir, 'flexible_dual_grid_cpu.cpp')]!r},\n"
                f"    include_dirs={[_OVOX_SRC, os.path.join(_OVOX_SRC, 'convert'), eigen]!r},\n"
                "    extra_compile_args=['-O3','-std=c++17','-Wno-c++11-narrowing','-Wno-sign-compare'],\n"
                ")], cmdclass={'build_ext': BuildExtension})\n"
            )
        proc = subprocess.run(
            [os.sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=build_dir, capture_output=not verbose, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:] if not verbose else ""
            raise RuntimeError(f"ovox_cpu build failed:\n{tail}")
        existing = glob.glob(os.path.join(build_dir, "ovox_cpu*.so"))
        if not existing:
            raise RuntimeError(f"ovox_cpu build produced no .so in {build_dir}")

    spec = importlib.util.spec_from_file_location("ovox_cpu", existing[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(eigen_include: Optional[str] = None, build_dir: Optional[str] = None, verbose: bool = False):
    """JIT-compile the CPU voxelizer and install it as o_voxel's `_C`.

    `eigen_include` must be a directory containing `Eigen/Dense`. Falls back to
    $EIGEN_INCLUDE_DIR, then to common Homebrew locations.
    """
    global _module
    if _module is not None:
        return _module



    eigen = eigen_include or os.environ.get("EIGEN_INCLUDE_DIR")
    if eigen is None:
        for cand in (
            os.path.join(_REPO, "TRELLIS.2", "o-voxel", "third_party", "eigen"),
            "/opt/homebrew/include/eigen3",
            "/usr/local/include/eigen3",
            "/usr/include/eigen3",
        ):
            if os.path.exists(os.path.join(cand, "Eigen", "Dense")):
                eigen = cand
                break
    if eigen is None or not os.path.exists(os.path.join(eigen, "Eigen", "Dense")):
        raise RuntimeError(
            "Eigen headers not found. o-voxel's third_party/eigen submodule is "
            "not initialised here. Pass eigen_include=... or set "
            "EIGEN_INCLUDE_DIR to a directory containing Eigen/Dense "
            "(`brew install eigen`, or unpack any eigen-3.4.x source tree)."
        )

    build_dir = build_dir or os.path.join(_REPO, ".build", "ovox_cpu")
    os.makedirs(build_dir, exist_ok=True)
    _patch_source(build_dir)

    _module = _build_with_setuptools(build_dir, eigen, verbose)

    # o_voxel's python wrapper dispatches to a module-global `_C`; it is never
    # bound because the packaged extension failed to build. Injecting here lets
    # the upstream wrapper (argument marshalling, aabb handling) be used as-is
    # rather than reimplemented.
    import o_voxel.convert.flexible_dual_grid as fdg
    fdg._C = _module
    return _module


def mesh_to_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int = 512,
    *,
    face_weight: float = 1.0,
    boundary_weight: float = 0.2,
    regularization_weight: float = 1e-2,
    eigen_include: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mesh -> (voxel_indices, local_offsets, intersected), encoder-ready.

    The mesh must already be normalised into the [-0.5, 0.5] cube.

    Returns `local_offsets` in [0, 1] **relative to each voxel**, not the
    global grid coordinates the raw o-voxel call returns. The encoder's
    `forward` does `vertices.feats - 0.5`, so it expects the local convention
    (verified empirically: raw dual_vertices come back in [0,1] over the whole
    grid, and `dv * resolution - voxel_indices` lands in [0,1] with mean 0.489).
    Passing the raw values straight through is silently wrong -- it looks like a
    working encode and produces garbage latents.
    """
    build(eigen_include=eigen_include)
    import o_voxel.convert.flexible_dual_grid as fdg

    voxel_indices, dual_vertices, intersected = fdg.mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=face_weight,
        boundary_weight=boundary_weight,
        regularization_weight=regularization_weight,
    )
    local = (dual_vertices * resolution - voxel_indices).clamp(0.0, 1.0)
    return voxel_indices, local, intersected


def normalize_mesh(mesh):
    """Centre and scale a trimesh into the [-0.5, 0.5] cube, in place.

    0.99999 rather than 1.0 for the same reason o-voxel's own example uses it:
    a vertex exactly on the grid boundary lands on voxel index `resolution`,
    one past the end.
    """
    b = mesh.bounding_box.bounds
    mesh.apply_translation(-(b[0] + b[1]) / 2)
    mesh.apply_scale(0.99999 / (b[1] - b[0]).max())
    return mesh
