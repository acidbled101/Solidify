"""
Environment/sys.path setup that MUST run before any TRELLIS or torch import.

MPS fallback MUST be set before torch is imported anywhere (including
transitively via flex_gemm). Without this, segment_reduce and a few other
ops crash instead of falling back to CPU.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_configured = False


def configure_env_and_paths():
    """Idempotent: safe to call more than once (uses setdefault throughout)."""
    global _configured
    if _configured:
        return

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

    try:
        import flex_gemm  # noqa: F401
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        # ImportError: package not installed (SKIP_METAL=1 or install failed).
        # RuntimeError: metallib load failure -- flex_gemm ships an MSL 4.0
        # metallib which only loads on macOS 26+. On older macOS the import
        # itself raises "Failed to load metallib ... language version 4.0
        # which is not supported on this OS". Fall back to conv_none either way.
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

    # stubs/ is appended (not prepended) so a pip-installed o_voxel wins over
    # our package stub -- the flat override module o_voxel_override_convert
    # is still discoverable either way because it doesn't collide with any
    # package.
    trellis_dir = os.path.join(_REPO_ROOT, "TRELLIS.2")
    stubs_dir = os.path.join(_REPO_ROOT, "stubs")
    if trellis_dir not in sys.path:
        sys.path.insert(0, trellis_dir)
    if stubs_dir not in sys.path:
        sys.path.append(stubs_dir)
    # Repo root itself: lets modules resolve top-level packages regardless of
    # which script is the actual process entrypoint (e.g. when server/main.py
    # is the entrypoint, not generate.py/make_printable.py).
    if _REPO_ROOT not in sys.path:
        sys.path.append(_REPO_ROOT)
    # third_party/ holds the upstream Apple Silicon port (see THIRD_PARTY.md).
    # Putting it on the path rather than rewriting its imports is deliberate:
    # `import backends.texture_baker` keeps resolving exactly as upstream wrote
    # it, so those files stay byte-identical to what they were forked from and
    # `git diff -M` against the fork point stays empty.
    third_party_dir = os.path.join(_REPO_ROOT, "third_party")
    if third_party_dir not in sys.path:
        sys.path.append(third_party_dir)

    _configured = True


def is_flex_gemm_available():
    return os.environ.get("SPARSE_CONV_BACKEND") == "flex_gemm"
