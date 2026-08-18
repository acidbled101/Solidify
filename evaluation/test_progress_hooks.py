"""Verify the progress + preview wiring without loading a 1.3B model.

Both bugs that stopped previews appearing were wiring bugs, not model bugs:
an `os` name that was local to _run_job and therefore unbound inside the
callbacks, and a hook installed on a branch that only half the jobs take.
Neither needed a GPU to catch, and both cost a 25-minute job to find because
nothing exercised the wiring on its own. This does.

    python test_progress_hooks.py
"""

import os
import sys
import tempfile

sys.path.insert(0, ".")
from trellis_core import bootstrap  # noqa: F401
from trellis_core import progress


class FakeSampler:
    def sample_once(self, model, x_t, t, t_prev, cond=None, **kw):
        raise AssertionError("not exercised here")


class FakePipeline:
    """Only the surface `instrumented` touches."""

    def __init__(self, coords):
        self._coords = coords
        self.shape_slat_sampler = FakeSampler()
        self.shape_slat_normalization = {"mean": [0.0] * 32, "std": [1.0] * 32}
        self.calls = []

    def sample_sparse_structure(self, cond, resolution, num_samples=1, sampler_params={}):
        self.calls.append((resolution, num_samples))
        return self._coords


def fake_coords(n_side=12):
    """Shaped exactly like the real thing: torch.argwhere(...)[:, [0,2,3,4]].int()"""
    import numpy as np
    import torch

    g = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(g - (n_side - 1) / 2.0, axis=1)
    sel = g[(d > n_side * 0.30) & (d < n_side * 0.42)]
    batch = np.zeros((len(sel), 1))
    return torch.from_numpy(np.concatenate([batch, sel], 1)).int()


def main() -> int:
    import torch  # noqa: F401

    coords = fake_coords()
    pipe = FakePipeline(coords)
    out_dir = tempfile.mkdtemp(prefix="preview_test_")

    seen = {"structure": 0, "steps": [], "previews": []}

    # Mirror the worker's callbacks exactly, including calling os.* from inside
    # a closure -- which is what actually broke.
    def on_structure(c, grid):
        seen["structure"] += 1
        name = "preview_structure.glb"
        if progress.write_voxel_preview(c, os.path.join(out_dir, name), grid=grid):
            seen["previews"].append(name)

    def on_step(stage, step, total, overall):
        seen["steps"].append((stage, step, total, round(overall, 3)))

    ok = True
    with progress.instrumented(pipe, on_step=on_step, on_structure=on_structure,
                               expected_bars=3, preview_dir=out_dir):
        # 1) the structure hook must fire on the pipeline method itself, so it
        #    works no matter which generation path the job takes.
        got = pipe.sample_sparse_structure({"cond": None}, 32, 1, {})
        assert got is coords, "hook must pass the real coords through unchanged"

        # 2) progress must be reported for every sampler bar, monotonically.
        from trellis2.pipelines.samplers import flow_euler
        for desc in ("Sampling sparse structure", "Sampling shape SLat", "Sampling shape SLat"):
            for _ in flow_euler.tqdm(list(range(12)), desc=desc):
                pass

    # --- the hooks must be gone once the context exits ---------------------
    from trellis2.pipelines.samplers import flow_euler
    if flow_euler.tqdm.__name__ == "tqdm_hook":
        print("FAIL: tqdm was left patched"); ok = False
    if pipe.sample_sparse_structure.__name__ == "sample_structure_hook":
        print("FAIL: sample_sparse_structure was left patched"); ok = False

    print(f"structure hook fired : {seen['structure']}x")
    print(f"preview written      : {seen['previews']}")
    if seen["structure"] != 1:
        print("FAIL: structure hook did not fire exactly once"); ok = False
    if not seen["previews"]:
        print("FAIL: no preview file produced"); ok = False
    else:
        p = os.path.join(out_dir, seen["previews"][0])
        size = os.path.getsize(p)
        print(f"preview size         : {size/1024:.0f} KB")
        if size < 1024:
            print("FAIL: preview file is implausibly small"); ok = False

    overalls = [o for _, _, _, o in seen["steps"]]
    print(f"step reports         : {len(seen['steps'])} across "
          f"{len(set(s for s, *_ in seen['steps']))} stage name(s)")
    print(f"overall              : {overalls[0]} -> {overalls[-1]}")
    if overalls != sorted(overalls):
        print("FAIL: overall fraction went backwards (the cascade rewind bug)"); ok = False
    if not (0.9 < overalls[-1] < 1.0):
        print(f"FAIL: overall ended at {overalls[-1]}, expected just under 1.0"); ok = False

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
