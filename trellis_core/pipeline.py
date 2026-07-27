"""
Load the TRELLIS.2 pipeline and run image-to-mesh generation.

Shared by generate.py (CLI) and server/worker.py (web backend), so the
model-loading and generation code path only exists in one place.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from . import bootstrap  # noqa: F401  (ensures env/path setup already ran)

import torch
from PIL import Image as PILImage


class WatchdogError(RuntimeError):
    """Raised when generation output looks like it hit the known macOS GPU
    watchdog bug: a long-running Metal kernel gets silently killed, leaving
    empty tensors instead of raising a normal exception at the failure site.
    """

    def __init__(self, message, stage="generation", raw_message=None):
        super().__init__(message)
        self.stage = stage
        self.raw_message = raw_message if raw_message is not None else message


_WATCHDOG_SIGNATURES = (
    "non-zero size",
    "BVH needs at least 8 triangles",
)


def watchdog_help_message():
    return (
        "The decoder produced an empty mesh.\n"
        "On Apple Silicon this is almost always the macOS GPU watchdog\n"
        "killing a long-running Metal kernel in the SLat decoder. The Metal\n"
        "error prints to stderr above (look for\n"
        "'kIOGPUCommandBufferCallbackErrorImpactingInteractivity') but does\n"
        "not raise a Python exception, so execution continues with empty\n"
        "tensors and crashes downstream.\n"
        "\n"
        "Workarounds, cheapest first:\n"
        "  1. Run headless -- close the lid / unplug external displays and\n"
        "     re-run over SSH. The watchdog tightens with WindowServer load.\n"
        "  2. MTL_CAPTURE_ENABLED=1 python generate.py ...   (extends the\n"
        "     watchdog timeout as a side effect of Metal-debugger mode)\n"
        "  3. SPARSE_CONV_BACKEND=none python generate.py ... (slower path,\n"
        "     may not help if a single dispatch is the offender)\n"
        "\n"
        "Tracking issue: https://github.com/shivampkumar/trellis-mac/issues\n"
    )


def load_pipeline(model_id: str, device: str = "mps"):
    """Load and move the TRELLIS.2 pipeline onto `device`. Takes ~100s."""
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.to(torch.device(device))
    return pipeline


@dataclass
class GenerationResult:
    glb_path: str
    obj_path: str
    vertex_count: int
    face_count: int
    generation_seconds: float
    bake_seconds: float
    used_metal_bake: bool


def _run_geometry_only(pipeline, image, *, seed, pipeline_type, sampler_params):
    """Shape-only inference: mirror Trellis2ImageTo3DPipeline.run() but skip the
    texture-SLat sampling and texture decode.

    The mesh is produced entirely from the shape latent (`decode_shape_slat`);
    the texture latent only adds surface colour, which we discard under
    `no_texture` anyway. Because we replay exactly the same RNG sequence with the
    same seed (preprocess -> manual_seed -> get_cond -> sparse structure -> shape
    SLat), the shape latent -- and therefore the geometry -- is identical to a
    full run(); we simply never spend the texture diffusion phase.

    This lives here (our tracked code), not in the vendored, git-ignored
    TRELLIS.2 tree, and calls the pipeline's own methods so it stays in step with
    whatever model is installed. It intentionally mirrors run()'s branching; if
    upstream run() changes materially, revisit this.
    """
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}
    if pipeline_type not in ss_res:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    img = pipeline.preprocess_image(image)
    torch.manual_seed(seed)
    cond_512 = pipeline.get_cond([img], 512)
    cond_1024 = pipeline.get_cond([img], 1024) if pipeline_type != "512" else None

    coords = pipeline.sample_sparse_structure(cond_512, ss_res[pipeline_type], 1, sampler_params)

    models = pipeline.models
    if pipeline_type == "512":
        shape_slat = pipeline.sample_shape_slat(
            cond_512, models["shape_slat_flow_model_512"], coords, sampler_params
        )
        res = 512
    elif pipeline_type == "1024":
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, models["shape_slat_flow_model_1024"], coords, sampler_params
        )
        res = 1024
    elif pipeline_type == "1024_cascade":
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
            512, 1024, coords, sampler_params,
        )
    else:  # 1536_cascade
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
            512, 1536, coords, sampler_params,
        )

    meshes, _subs = pipeline.decode_shape_slat(shape_slat, res)
    mesh = meshes[0]
    mesh.fill_holes()  # same hole-fill run()'s decode_latent applies
    return mesh


def run_generation(
    pipeline,
    image: PILImage.Image,
    *,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    steps: Optional[int] = None,
    target_faces: int = 1000000,
    texture_size: int = 2048,
    no_texture: bool = False,
    skip_texture: bool = False,
    out_glb_path: str,
    out_obj_path: str,
) -> GenerationResult:
    """Run TRELLIS.2 generation + (optional) texture bake, writing GLB/OBJ.

    `skip_texture` skips the texture-SLat *inference* entirely (see
    `_run_geometry_only`): a full ~1.3B-param diffusion phase plus texture decode
    that produce colour we discard whenever `no_texture` is set. The mesh comes
    from the shape latent alone, so geometry is unchanged; only the wasted work
    goes away. Skipping texture inference means there is nothing to bake, so it
    implies `no_texture`.

    Raises WatchdogError if the output looks corrupted by the known macOS
    GPU-watchdog bug (see watchdog_help_message()).
    """
    if skip_texture:
        no_texture = True  # no texture was inferred, so there's nothing to bake

    sampler_overrides = {"steps": steps} if steps else {}

    t0 = time.time()
    try:
        if skip_texture:
            mesh_out = _run_geometry_only(
                pipeline, image, seed=seed, pipeline_type=pipeline_type,
                sampler_params=sampler_overrides,
            )
        else:
            outputs = pipeline.run(
                image,
                seed=seed,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params=sampler_overrides,
                shape_slat_sampler_params=sampler_overrides,
                tex_slat_sampler_params=sampler_overrides,
            )
            mesh_out = outputs[0] if isinstance(outputs, list) else outputs
    except (IndexError, AssertionError) as e:
        msg = str(e)
        if any(sig in msg for sig in _WATCHDOG_SIGNATURES):
            raise WatchdogError(watchdog_help_message(), raw_message=msg) from e
        raise
    t_gen = time.time() - t0

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise WatchdogError(watchdog_help_message(), raw_message="empty mesh (0 vertices/faces)")

    has_voxels = hasattr(mesh_out, "attrs") and mesh_out.attrs is not None
    tex_size = texture_size

    used_metal_bake = False
    t_bake0 = time.time()

    if has_voxels and not no_texture:
        try:
            import o_voxel.postprocess
            backend = getattr(o_voxel.postprocess, "_BACKEND", None)
            has_dr = getattr(o_voxel.postprocess, "_HAS_DR", False)
            use_metal = backend == "metal" and has_dr
            if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
                # o_voxel's _grid_sample_3d fallback returns [B*C, M] but the
                # bake consumes it as [M, C]. Patch it to transpose before the
                # reshape. We avoid installing flex_gemm itself because its
                # import slows the diffusion hot path ~10x on MPS.
                import torch.nn.functional as _F_gs

                def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
                    B, C = shape[0], shape[1]
                    D, H, W = shape[2], shape[3], shape[4]
                    device = feats.device
                    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
                    batch_idx = coords[:, 0].long()
                    cx = coords[:, 1].long(); cy = coords[:, 2].long(); cz = coords[:, 3].long()
                    dense_vol[batch_idx, :, cx, cy, cz] = feats
                    grid_norm = torch.stack([
                        grid[..., 2] / (W - 1) * 2 - 1,
                        grid[..., 1] / (H - 1) * 2 - 1,
                        grid[..., 0] / (D - 1) * 2 - 1,
                    ], dim=-1).reshape(B, 1, 1, -1, 3)
                    sampled = _F_gs.grid_sample(
                        dense_vol, grid_norm, mode="bilinear",
                        align_corners=True, padding_mode="border",
                    )
                    M = grid.shape[1]
                    return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

                o_voxel.postprocess._grid_sample_3d = _gs3d_fix
        except (ImportError, AttributeError):
            use_metal = False

        if use_metal:
            try:
                import o_voxel
                import fast_simplification

                verts_np = mesh_out.vertices.cpu().numpy()
                faces_np = mesh_out.faces.cpu().numpy()
                tf = min(target_faces, len(faces_np))
                if len(faces_np) > tf:
                    ratio = 1.0 - (tf / len(faces_np))
                    simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
                    simp_verts_t = torch.from_numpy(simp_verts).float().to(mesh_out.vertices.device)
                    simp_faces_t = torch.from_numpy(simp_faces.astype("int32")).to(mesh_out.faces.device)
                else:
                    simp_verts_t = mesh_out.vertices
                    simp_faces_t = mesh_out.faces

                glb = o_voxel.postprocess.to_glb(
                    vertices=simp_verts_t.cpu(),
                    faces=simp_faces_t.cpu(),
                    attr_volume=mesh_out.attrs.cpu(),
                    coords=mesh_out.coords.cpu(),
                    attr_layout=mesh_out.layout,
                    voxel_size=mesh_out.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=tf,
                    texture_size=tex_size,
                    verbose=True,
                )
                glb.export(out_glb_path)
                used_metal_bake = True
            except RuntimeError:
                used_metal_bake = False

        if not used_metal_bake:
            from backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

            voxel_coords = mesh_out.coords.cpu().float()
            voxel_attrs = mesh_out.attrs.cpu().float()
            origin = mesh_out.origin.cpu().float()
            vs = mesh_out.voxel_size

            bake_verts, bake_faces = verts, faces
            tf = min(target_faces, len(faces))
            if len(faces) > tf:
                try:
                    import fast_simplification
                    ratio = 1.0 - (tf / len(faces))
                    bake_verts, bake_faces = fast_simplification.simplify(verts, faces, ratio)
                except ImportError:
                    pass

            new_verts, new_faces, uvs, vmapping = uv_unwrap(bake_verts, bake_faces)

            base_color_img, mr_img, mask = bake_texture(
                new_verts, new_faces, uvs,
                voxel_coords.numpy(), voxel_attrs.numpy(),
                origin.numpy(), vs,
                texture_size=tex_size,
            )

            basecolor_path = os.path.splitext(out_glb_path)[0] + "_basecolor.png"
            PILImage.fromarray(base_color_img).save(basecolor_path)
            export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, out_glb_path)
    else:
        import trimesh

        # Apply the same target_faces cap the textured path already applies
        # (fast_simplification, see the branch above) -- without it, a complex
        # input photo can decode to a multi-million-face raw mesh that's both
        # slow to export and, worse, slow for the downstream make_printable
        # stage to repair/voxelize. Tested against a real 7.4M-face job output:
        # capping first cut that stage from 746s to 68s with no measurable
        # quality loss (near-identical Chamfer/Hausdorff fidelity), since
        # make_printable's own voxel remesh + resimplify is what determines
        # final quality either way.
        out_verts, out_faces = verts, faces
        tf = min(target_faces, len(faces))
        if len(faces) > tf:
            try:
                import fast_simplification
                ratio = 1.0 - (tf / len(faces))
                out_verts, out_faces = fast_simplification.simplify(verts, faces, ratio)
            except ImportError:
                pass

        tm = trimesh.Trimesh(vertices=out_verts, faces=out_faces)
        tm.export(out_glb_path)
        verts, faces = out_verts, out_faces  # keep OBJ export + result stats in sync

    t_bake = time.time() - t_bake0

    with open(out_obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return GenerationResult(
        glb_path=out_glb_path,
        obj_path=out_obj_path,
        vertex_count=int(verts.shape[0]),
        face_count=int(faces.shape[0]),
        generation_seconds=t_gen,
        bake_seconds=t_bake,
        used_metal_bake=used_metal_bake,
    )
