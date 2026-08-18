"""
Shared GLB load/export/simplify helpers used by simplify_glb.py and
make_printable.py. No TRELLIS/torch imports -- kept lightweight so these
tools work standalone.
"""

import numpy as np
import trimesh


def load_glb(path):
    """Load a .glb as a single trimesh.Trimesh, preserving vertex order."""
    scene_check = trimesh.load(path)
    if isinstance(scene_check, trimesh.Scene) and len(scene_check.geometry) > 1:
        print(
            f"  Warning: {path} has {len(scene_check.geometry)} mesh nodes; "
            "merging into one mesh with a single material (best-effort)."
        )

    # force='mesh' normalizes Scene -> Trimesh (even single-node glbs load as
    # a Scene). process=False keeps vertex indices aligned with visual.uv /
    # visual.vertex_colors -- default processing can dedupe/reorder vertices
    # and silently desync those arrays.
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.faces.shape[1] != 3:
        raise ValueError(f"Expected triangulated mesh, got faces of shape {mesh.faces.shape}")
    return mesh


def extract_visual_info(mesh):
    """Classify and extract a mesh's visual data into a normalized dict."""
    info = {
        "kind": "none",
        "uv": None,
        "base_color_img": None,
        "mr_img": None,
        "vertex_colors": None,
    }

    visual = mesh.visual
    if isinstance(visual, trimesh.visual.texture.TextureVisuals):
        uv = visual.uv
        material = visual.material
        base_color_img = getattr(material, "baseColorTexture", None)
        mr_img = getattr(material, "metallicRoughnessTexture", None)
        if uv is not None and base_color_img is not None:
            info["kind"] = "texture"
            info["uv"] = np.asarray(uv)
            info["base_color_img"] = base_color_img
            info["mr_img"] = mr_img
            return info
        # TextureVisuals without usable UV/texture -- fall through below.

    if isinstance(visual, trimesh.visual.color.ColorVisuals) or info["kind"] == "none":
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors)
            if len(np.unique(vertex_colors, axis=0)) > 1:
                info["kind"] = "vertex_color"
                info["vertex_colors"] = vertex_colors

    return info


def simplify_with_attrs(vertices, faces, target_faces, agg=7.0):
    """
    Decimate a mesh with fast_simplification, returning an index mapping
    from original vertices to decimated vertices so callers can resample
    per-vertex attributes (UV, vertex colors) consistently.

    Returns (new_verts, new_faces, indice_mapping_or_None). indice_mapping
    is None when no simplification was needed (target_faces >= n_faces).
    """
    import fast_simplification

    n_faces_in = len(faces)
    if n_faces_in <= target_faces:
        return vertices, faces, None

    new_verts, new_faces, collapses = fast_simplification.simplify(
        vertices, faces,
        target_count=target_faces,
        agg=agg,
        return_collapses=True,
    )

    # replay_simplification requires float32 points (strict dtype check),
    # even though simplify() itself accepts float64. Its own returned
    # points/faces are numerically identical (float32-rounded) to
    # new_verts/new_faces above and are discarded here -- we only need
    # indice_mapping: indice_mapping[i] = j means original vertex i maps
    # to decimated vertex j.
    _, _, indice_mapping = fast_simplification.replay_simplification(
        vertices.astype(np.float32), faces, collapses,
    )
    return new_verts, new_faces, indice_mapping


def resample_per_vertex(data, indice_mapping, n_out):
    """
    Resample a per-original-vertex attribute array into decimated vertex
    space using indice_mapping (indice_mapping[i] = j). "First writer wins"
    per decimated vertex; any decimated vertex with zero contributors falls
    back to nearest-neighbor fill from the vertices that were filled.
    """
    data = np.asarray(data)
    out = np.zeros((n_out,) + data.shape[1:], dtype=data.dtype)
    filled = np.zeros(n_out, dtype=bool)

    for orig_idx, dec_idx in enumerate(indice_mapping):
        if not filled[dec_idx]:
            out[dec_idx] = data[orig_idx]
            filled[dec_idx] = True

    if not filled.all():
        missing = np.where(~filled)[0]
        print(f"  Warning: {len(missing)} decimated vertices had no source attribute data; "
              f"filling with mean of resolved vertices")
        if filled.any():
            out[missing] = out[filled].mean(axis=0)

    return out


def export_mesh(vertices, faces, visual_info, output_path, new_uv=None, new_vertex_colors=None):
    """Export a mesh, dispatching on visual_info['kind']."""
    kind = visual_info.get("kind", "none")

    if kind == "texture" and new_uv is not None:
        from backends.texture_baker import export_glb_with_texture

        base_color_arr = np.array(visual_info["base_color_img"])
        mr_img = visual_info.get("mr_img")
        mr_arr = np.array(mr_img) if mr_img is not None else None
        return export_glb_with_texture(vertices, faces, new_uv, base_color_arr, mr_arr, output_path)

    if kind == "vertex_color" and new_vertex_colors is not None:
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=new_vertex_colors, process=False)
        tm.export(output_path)
        return output_path

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.export(output_path)
    return output_path
