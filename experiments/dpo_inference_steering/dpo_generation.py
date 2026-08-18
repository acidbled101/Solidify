"""
Phase 4 of experiments/dpo_inference_steering/DESIGN_NOTE.md: end-to-end generation through the
Phase 2/3 DPO-branched shape-SLat sampling, handing the result to the
EXISTING, unmodified print-prep pipeline in trellis_core/printprep/printable.py
(repair_watertight -> diagnostics -> fidelity -> export_glb_and_stl) rather
than reimplementing any of that.

Scope: GEOMETRY ONLY -- this never samples a texture SLat at all (see
`textured=False` on DPOGenerationResult), not merely "decodes texture and
throws it away". Texture baking is a separate, already-working concern
handled by trellis_core/pipeline.py's run_generation() (Metal o_voxel bake,
or the xatlas/KDTree fallback); reimplementing that here would duplicate a
lot of already-shipped, tested logic for something Phase 4 doesn't need
(making the shape printable, not how it looks), and --solid-infill (on by
default in PrintableOptions) voxel-remeshes anyway, which
trellis_core/printprep/printable.py's repair_watertight() already documents as
discarding the original topology/UV atlas -- so a full texture bake would
mostly be thrown away downstream regardless. If a textured output from a
DPO-branched generation is wanted later, decode shape_slat/tex_slat and feed
them through pipeline.py's existing bake path instead of writing a second one.

Restricted to pipeline_type '512' and '1024' -- the only two paths that call
Trellis2ImageTo3DPipeline.sample_shape_slat, which
dpo_branch.sample_shape_slat_with_dpo_branch is a drop-in alternative for
(see that module's docstring). The default '1024_cascade' pipeline_type calls
a differently-shaped sample_shape_slat_cascade and is not supported here.

NOT VERIFIED ON REAL HARDWARE -- same caveat as dpo_branch.py: every call here
was written against the source in TRELLIS.2/ but never executed (no MPS
device or model weights on the machine this was written on).
"""

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from trellis_core import bootstrap  # noqa: F401  (ensures env/path setup already ran)

import torch
import trimesh
from PIL import Image as PILImage

from . import dpo_branch
from trellis_core.printprep import printable
from .pipeline import WatchdogError, watchdog_help_message, _WATCHDOG_SIGNATURES


@dataclass
class DPOGenerationResult:
    """Outcome of run_generation_with_dpo: the existing printable pipeline's
    result, plus the DPO branch step's own report and raw (pre-repair) mesh
    stats.

    textured is always False -- this module never samples a texture SLat at
    all (see module docstring). Present so a future CLI/server caller has a
    programmatic signal rather than having to know that from prose, unlike
    trellis_core/pipeline.py's GenerationResult which really does carry a
    texture (used_metal_bake tells you which bake path, not whether one ran).
    """
    printable_result: printable.PrintableResult
    dpo_report: Optional[dpo_branch.DPOMultiBranchReport]
    raw_vertex_count: int
    raw_face_count: int
    generation_seconds: float
    postprocess_seconds: float
    textured: bool = False


def run_generation_with_dpo(
    pipeline,
    image: PILImage.Image,
    *,
    seed: int = 42,
    pipeline_type: str = "512",
    steps: Optional[int] = None,
    dpo_config: Optional[dpo_branch.DPOBranchConfig] = None,
    output_prefix: str,
    target_faces: int = 1000000,
    overhang_angle: float = 45.0,
    solid_infill: bool = True,
    voxel_pitch: Optional[float] = None,
    min_object_ratio: float = 0.05,
    max_objects: int = 10,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> DPOGenerationResult:
    """Generate via the DPO-branched shape-SLat sampler, then run the result
    through printable.py's EXISTING single-object pipeline
    (repair_watertight -> diagnostics -> fidelity -> export), unmodified.

    Mirrors trellis_core/pipeline.py's run_generation() / TRELLIS.2's
    Trellis2ImageTo3DPipeline.run() for the cond / sparse-structure stages --
    same pipeline methods, same arguments, same order -- substituting
    dpo_branch.sample_shape_slat_with_dpo_branch for the ordinary
    pipeline.sample_shape_slat at the shape-SLat stage, and skipping the
    texture-SLat stage entirely (geometry only, see module docstring).

    Raises WatchdogError under the same conditions trellis_core/pipeline.py's
    run_generation() does -- both the empty-mesh case AND the
    IndexError/AssertionError signature match against _WATCHDOG_SIGNATURES,
    which can raise from inside sample_sparse_structure/decode_shape_slat
    before an empty-mesh check would ever be reached.

    on_event: forwarded verbatim to
        dpo_branch.sample_shape_slat_with_dpo_branch (see its docstring for
        the event types/payloads) -- this function additionally emits its
        own "stage" events around the parts it owns (cond/sparse-structure,
        decode, print-prep) and a final "printable_result" event carrying
        the existing printable.py pipeline's diagnostics/fidelity/watertight
        verdict, so a caller doesn't need a second mechanism to see the
        post-processing outcome. See experiments/dpo_inference_steering/inspector/ for the consumer.
    """
    def _emit(event_type: str, **payload) -> None:
        if on_event is None:
            return
        try:
            on_event(event_type, payload)
        except Exception as e:  # pragma: no cover - reporting must never break generation
            print(f"[dpo_generation] on_event callback raised ({event_type}): {e}")
    if pipeline_type not in ("512", "1024"):
        raise ValueError(
            f"run_generation_with_dpo only supports pipeline_type '512' or "
            f"'1024' (got {pipeline_type!r}) -- the '1024_cascade' default "
            f"used elsewhere in this repo (see generate.py) calls a "
            f"differently-shaped sample_shape_slat_cascade that dpo_branch.py "
            f"does not support yet (see that module's docstring)."
        )

    for key in (f"shape_slat_flow_model_{pipeline_type}",):
        assert key in pipeline.models, (
            f"No {pipeline_type} resolution shape SLat flow model found "
            f"(missing pipeline.models[{key!r}])."
        )

    resolution = int(pipeline_type)
    dpo_config = dpo_config or dpo_branch.DPOBranchConfig()
    if dpo_config.decode_resolution != resolution:
        # DPOBranchConfig's own default (512) silently mismatches a '1024'
        # pipeline_type: TRELLIS.2's coords/resolution relation is pinned at
        # 16x (ss_res 32<->decode 512, ss_res 64<->decode 1024 -- see
        # trellis2_image_to_3d.py's hr_resolution // 16 and this function's
        # own ss_res mapping below), so decoding at the wrong resolution
        # doesn't error, it silently mis-scales every vertex (a '1024' mesh
        # decoded at decode_resolution=512 comes out ~2x oversized and
        # translated out of the unit cube), which then makes every
        # geometric_judge verdict at the branch point meaningless. This is
        # NOT a caller-overridable choice -- decode_resolution is fully
        # determined by pipeline_type, so a mismatched explicit dpo_config
        # gets overridden unconditionally (with a loud print, not silently),
        # the same as one built with the wrong default.
        print(
            f"[dpo_generation] dpo_config.decode_resolution="
            f"{dpo_config.decode_resolution} does not match pipeline_type="
            f"{pipeline_type!r}'s required resolution ({resolution}) -- "
            f"overriding to {resolution}."
        )
        dpo_config = dataclasses.replace(dpo_config, decode_resolution=resolution)

    sampler_overrides = {"steps": steps} if steps else {}
    t0 = time.time()

    try:
        _emit("stage", name="preprocess_image")
        image = pipeline.preprocess_image(image)
        torch.manual_seed(seed)
        # sample_sparse_structure ALWAYS takes the 512 cond, even on the
        # '1024' path -- matching Trellis2ImageTo3DPipeline.run()'s
        # cond_512/cond_1024 split (run() computes both up front and always
        # feeds cond_512 to sample_sparse_structure; cond_1024 -- used here
        # only when pipeline_type == '1024' -- feeds the shape-SLat stage).
        # Collapsing this into a single "cond_resolution" that also drove
        # sample_sparse_structure was a real bug caught in review: a '1024'
        # cond is a 64x64 DINOv3 patch grid the sparse-structure model was
        # never trained on, and it fails silently (no positional embedding on
        # cond tokens to raise a shape mismatch on) rather than erroring.
        cond_512 = pipeline.get_cond([image], 512)
        cond = cond_512 if pipeline_type == "512" else pipeline.get_cond([image], 1024)

        _emit("stage", name="sample_sparse_structure")
        ss_res = {"512": 32, "1024": 64}[pipeline_type]
        coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, sampler_overrides)
        _emit("stage", name="sparse_structure_done", n_voxels=int(coords.shape[0]))

        flow_model = pipeline.models[f"shape_slat_flow_model_{pipeline_type}"]
        shape_slat, dpo_report = dpo_branch.sample_shape_slat_with_dpo_branch(
            pipeline, cond, flow_model, coords,
            sampler_params=sampler_overrides,
            dpo_config=dpo_config,
            return_report=True,
            on_event=on_event,
        )

        # torch.cuda.is_available() is always False on Apple Silicon -- there
        # is no CUDA device -- so this call never actually fired on the target
        # hardware. MPS has its own cache to release; without this, cached
        # blocks freed by dpo_branch.py's `del`s further up were never
        # actually returned to the pool before the (memory-heavier) decode
        # below ran.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # decode_shape_slat directly, NOT decode_latent(shape_slat, tex_slat,
        # ...) -- decode_latent requires a tex_slat, which would mean paying
        # for a full texture-SLat sampling stage only to discard it a few
        # lines later (geometry only, see module docstring). decode_latent's
        # only other per-mesh effect is calling m.fill_holes(), which
        # third_party/patches/mps_compat.py's patch_mesh_base() has already turned into an
        # unconditional no-op on this repo's patched TRELLIS.2 checkout (the
        # Metal cumesh port segfaults on decoder-sized meshes) -- so skipping
        # decode_latent loses nothing beyond the tex stage itself.
        _emit("stage", name="decode_shape_slat")
        meshes, _subs = pipeline.decode_shape_slat(shape_slat, resolution)  # already @torch.no_grad()
    except (IndexError, AssertionError) as e:
        msg = str(e)
        if any(sig in msg for sig in _WATCHDOG_SIGNATURES):
            raise WatchdogError(watchdog_help_message(), raw_message=msg) from e
        raise

    if not meshes:
        raise WatchdogError(watchdog_help_message(), raw_message="empty mesh (no meshes returned)")
    mesh_out = meshes[0]
    verts = mesh_out.vertices.detach().cpu().numpy()
    faces = mesh_out.faces.detach().cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise WatchdogError(watchdog_help_message(), raw_message="empty mesh (0 vertices/faces)")

    t_gen = time.time() - t0
    t_post0 = time.time()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Drop floating-speck noise before repair, matching
    # printable.run_make_printable's single-object path (which this function
    # otherwise reimplements the body of) -- a raw decoder mesh routinely
    # carries disconnected specks, and process_object derives its voxel pitch
    # from the mesh's OWN bounding-box diagonal, so a single distant speck
    # would coarsen the solidification grid for the whole model.
    comps = printable.significant_components(mesh, min_ratio=min_object_ratio, max_objects=max_objects)
    if len(comps) > 1 and dpo_config.verbose:
        print(f"[dpo_generation] {len(comps)} large objects detected; keeping the largest only.")
    mesh = comps[0]
    original = mesh.copy()
    # The mesh as decoded, before printable.process_object()'s watertight
    # repair / decimation / solid-infill remesh -- exposed so a caller (see
    # experiments/dpo_inference_steering/inspector/) can compare the DPO-branched result against a reference
    # without print-prep's own resolution/topology changes in the way.
    _emit("raw_result", mesh=original)

    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)

    _emit("stage", name="print_prep", raw_vertex_count=int(verts.shape[0]), raw_face_count=int(faces.shape[0]))
    opts = printable.PrintableOptions(
        target_faces=target_faces, solid_infill=solid_infill, voxel_pitch=voxel_pitch,
    )
    processed, visual_info, new_vertex_colors = printable.process_object(mesh, opts)
    fid = printable.fidelity(original, processed)
    diag = printable.diagnostics(processed, overhang_angle)
    glb_path, stl_path = printable.export_glb_and_stl(processed, visual_info, new_vertex_colors, output_prefix)

    t_postprocess = time.time() - t_post0
    _emit(
        "printable_result",
        watertight=bool(processed.is_watertight), diagnostics=diag, fidelity=fid,
        vertex_count=int(len(processed.vertices)), face_count=int(len(processed.faces)),
        generation_seconds=t_gen, postprocess_seconds=t_postprocess,
        glb_path=glb_path, stl_path=stl_path,
    )

    printable_result = printable.PrintableResult(
        mode="single",
        files=[{"kind": "printable_glb", "path": glb_path}, {"kind": "printable_stl", "path": stl_path}],
        diagnostics=diag,
        fidelity=fid,
        watertight=bool(processed.is_watertight),
        seconds=t_postprocess,
        glb_path=glb_path,
        stl_path=stl_path,
    )

    return DPOGenerationResult(
        printable_result=printable_result,
        dpo_report=dpo_report,
        raw_vertex_count=int(verts.shape[0]),
        raw_face_count=int(faces.shape[0]),
        generation_seconds=t_gen,
        postprocess_seconds=t_postprocess,
    )
