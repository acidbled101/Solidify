"""Real progress and live shape previews from a running TRELLIS.2 generation.

WHY THIS EXISTS
---------------
The web UI used to fake its progress bar: a curve fitted to elapsed time, which
drifts badly because generation time varies with the number of occupied voxels
(measured range on real photos: 114s to 244s for the same settings). A user
watching 60% for two minutes has no idea whether anything is happening.

The samplers already know exactly where they are -- every one of them drives a
tqdm bar. This taps those bars and turns them into real progress, then goes one
better and shows the actual shape as it forms.

HOW THE PREVIEW IS ALMOST FREE
------------------------------
FlowEulerSampler.sample_once returns `pred_x_0` at every step: the model's
current estimate of the FINISHED latent, not the noisy intermediate. It is
already computed and already stored (`ret.pred_x_0` in the sampler). So a
preview costs no extra sampling at all -- only a decode, and only at whichever
steps we ask for.

The decode is the expensive half of generation, so previews are decoded at a
much lower resolution than the final mesh and only at a couple of points.
Measure before raising either: see `bench_preview.py`.

WHY MONKEYPATCHING, NOT A FORK
------------------------------
trellis2 is vendored and git-ignored. Editing it means the change is invisible
to review and vanishes on reinstall. Both hooks here are installed and removed
by a context manager, touch only two named attributes, and restore them in a
`finally` -- so a crash mid-generation cannot leave the pipeline patched.

The tqdm patch is applied to `flow_euler`'s module-level name only, so it
affects the sampling loops and nothing else that uses tqdm (weight loading, for
instance, keeps its own bar).
"""

import contextlib
import logging
import os
import time
from typing import Callable, Optional, Sequence

log = logging.getLogger("trellis.progress")

# tqdm_desc strings the vendored pipeline passes, mapped to the stage names the
# UI shows. Matching on a substring keeps this working if upstream reworded.
STAGE_LABELS = (
    ("sparse structure", "structure"),
    ("shape slat", "shape"),
    ("texture slat", "texture"),
)


def stage_of(desc: str) -> str:
    d = (desc or "").lower()
    for needle, label in STAGE_LABELS:
        if needle in d:
            return label
    return "sampling"


@contextlib.contextmanager
def instrumented(
    pipeline,
    *,
    on_step: Optional[Callable[[str, int, int, float], None]] = None,
    on_preview: Optional[Callable[[str, int], None]] = None,
    expected_bars: int = 3,
    preview_at: Sequence[float] = (),
    preview_resolution: int = 128,
    preview_dir: Optional[str] = None,
):
    """Report sampler progress, and optionally write preview meshes mid-flight.

    on_step(stage, step, total, overall) -- once per sampling step, every stage.
                                     `overall` is the monotonic 0..1 fraction
                                     across the whole generation.
    on_preview(path, step)        -- called after a preview GLB is written.
    expected_bars                 -- how many tqdm bars this pipeline_type
                                     produces; 3 for a cascade (structure, then
                                     the shape sampler twice), 2 otherwise.
    preview_at                    -- fractions of the SHAPE stage at which to
                                     DECODE a preview. Empty disables them and
                                     removes that hook. Off by default: measured
                                     at ~70s per decode even at resolution 128,
                                     11.8% of a run. write_voxel_preview is the
                                     cheap alternative.
    """
    from trellis2.pipelines.samplers import flow_euler

    # Shared between the two hooks: the tqdm wrapper knows which stage and step
    # we are on, sample_once knows the latent. Neither knows both.
    cur = {"stage": "", "step": 0, "total": 0, "bar": -1, "fired": set()}

    real_tqdm = flow_euler.tqdm

    def overall(bar, step, total):
        """Fraction across the whole generation, never decreasing.

        A cascade runs the shape sampler twice, so step/total alone rewinds to
        zero partway through and the bar would visibly jump backwards. Counting
        completed bars against the number expected for this pipeline_type keeps
        it monotonic.

        If more bars appear than expected -- an upstream change, or a pipeline
        type we did not anticipate -- this saturates near the end instead of
        exceeding 1.0 or resetting.
        """
        if expected_bars <= 0:
            return 0.0
        within = (step / total) if total else 0.0
        return min(0.999, (bar + within) / expected_bars)

    def tqdm_hook(iterable=None, **kw):
        if iterable is None:                      # not a form the samplers use
            return real_tqdm(iterable, **kw)
        try:
            total = kw.get("total") or len(iterable)
        except TypeError:
            total = 0
        stage = stage_of(kw.get("desc", ""))
        cur["bar"] += 1
        cur.update(stage=stage, total=total, step=0)
        bar = cur["bar"]

        def report(step):
            if not on_step:
                return
            try:
                on_step(stage, step, total, overall(bar, step, total))
            except Exception:                     # a reporting bug must never
                log.debug("on_step failed", exc_info=True)   # kill a job

        def gen():
            for i, item in enumerate(iterable):
                cur["step"] = i
                report(i)
                yield item
            cur["step"] = total
            report(total)

        return gen()

    sampler = pipeline.shape_slat_sampler
    real_sample_once = sampler.sample_once
    want_previews = bool(preview_at) and preview_dir is not None

    def sample_once_hook(model, x_t, t, t_prev, cond=None, **kw):
        out = real_sample_once(model, x_t, t, t_prev, cond, **kw)
        if cur["stage"] == "shape" and cur["total"]:
            frac = (cur["step"] + 1) / cur["total"]
            for target in preview_at:
                if target not in cur["fired"] and frac >= target:
                    cur["fired"].add(target)
                    _write_preview(pipeline, out.pred_x_0, cur["step"],
                                   preview_resolution, preview_dir, on_preview)
        return out

    flow_euler.tqdm = tqdm_hook
    if want_previews:
        sampler.sample_once = sample_once_hook
    try:
        yield cur
    finally:
        flow_euler.tqdm = real_tqdm
        if want_previews:
            sampler.sample_once = real_sample_once


def write_voxel_preview(coords, out_path: str, grid: int = 32) -> bool:
    """Export the sparse structure as a box mesh. This is the cheap preview.

    MEASURED: decoding an in-flight latent with the real shape decoder costs
    ~70s even at resolution 128 -- 11.8% added to a 1328s generation -- because
    the cost is dominated by the sparse convolution over occupied voxels, which
    does not shrink when you lower the output resolution. Two previews bought
    two minutes of GPU time. That is not a preview, it is a tax.

    The structure stage has already produced `coords`: the occupied voxels of
    the object, typically a few thousand of them. Turning those into boxes is
    pure CPU work, takes well under a second, and is a genuine picture of the
    shape -- blocky, but the real silhouette, available minutes before the mesh.

    What it cannot do is animate during shape sampling: coords are fixed once
    the structure stage ends, and only the per-voxel latent features change
    after that. So this is one preview, shown early, not a forming animation.
    """
    import numpy as np
    import trimesh

    try:
        c = coords.detach().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
        if c.ndim != 2 or c.shape[0] == 0:
            return False
        xyz = c[:, 1:4] if c.shape[1] >= 4 else c[:, :3]      # drop the batch column
        xyz = xyz.astype(np.float32)

        # Centre on the origin and fit in a unit cube, matching the convention
        # the final mesh uses so the viewer camera does not have to change.
        centred = (xyz + 0.5) / float(grid) - 0.5
        pitch = 1.0 / float(grid)

        box = trimesh.creation.box(extents=(pitch, pitch, pitch))
        bv, bf = box.vertices, box.faces
        n = len(centred)
        verts = (bv[None, :, :] + centred[:, None, :]).reshape(-1, 3)
        faces = (bf[None, :, :] + (np.arange(n) * len(bv))[:, None, None]).reshape(-1, 3)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        # file_type is explicit: the temp name ends in .tmp, and trimesh infers
        # the exporter from the extension, so it would raise "exporter not
        # available" for a perfectly good mesh.
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(tmp, file_type="glb")
        os.replace(tmp, out_path)
        log.info("voxel preview: %d voxels -> %s", n, os.path.basename(out_path))
        return True
    except Exception:
        log.debug("voxel preview failed", exc_info=True)
        return False


def _write_preview(pipeline, pred_x_0, step, resolution, out_dir, on_preview):
    """Decode one in-flight latent to a coarse GLB.

    Every failure here is swallowed. A preview is a nicety; a partially denoised
    latent can legitimately decode to nothing, and early steps routinely produce
    a mesh too small for the decoder's BVH ("BVH needs at least 8 triangles" is
    one of the known watchdog signatures). None of that is a reason to fail a
    generation the user is waiting on.
    """
    import torch
    import trimesh

    t0 = time.time()
    try:
        norm = pipeline.shape_slat_normalization
        std = torch.tensor(norm["std"])[None].to(pred_x_0.device)
        mean = torch.tensor(norm["mean"])[None].to(pred_x_0.device)
        slat = pred_x_0 * std + mean

        with torch.no_grad():
            meshes, _ = pipeline.decode_shape_slat(slat, resolution)
        m = meshes[0]
        v = m.vertices.detach().float().cpu().numpy()
        f = m.faces.detach().cpu().numpy()
        if len(f) < 8:
            log.debug("preview at step %d had %d faces -- skipped", step, len(f))
            return

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"preview_{step:03d}.glb")
        tmp = path + ".tmp"
        trimesh.Trimesh(vertices=v, faces=f, process=False).export(tmp, file_type="glb")
        os.replace(tmp, path)                     # never serve a half-written file

        log.info("preview step %d: %d faces at res %d in %.1fs",
                 step, len(f), resolution, time.time() - t0)
        if on_preview:
            with contextlib.suppress(Exception):
                on_preview(path, step)
    except Exception:
        log.debug("preview at step %d failed", step, exc_info=True)
