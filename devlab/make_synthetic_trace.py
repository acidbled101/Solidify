"""
Builds a synthetic-but-real trace for UI verification without any GPU work.

"Synthetic" only in that there's no real DPO branching happening -- the
sampler schedule is the REAL build_t_pairs(24) output, the branch windows
come from the REAL DPOBranchConfig.effective_t_branches()/
select_branch_windows() (num_branches=2, so this trace also exercises the
multi-branch path, not just the original single-fork one), the candidate
scores are REAL geometric_judge.score_mesh_detailed() calls on real trimesh
geometry (so JudgeScore/JudgeDetails serialization, histogram binning, and
mesh export all go through the exact same code a live run uses), and branch
0's gradient-step numbers (loss/proxy/rms) are the actual values captured
from this session's real hardware smoke test of dpo_branch.py, not invented.

Run: python devlab/make_synthetic_trace.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from devlab.trace import TraceWriter
from trellis_core import geometric_judge as gj
from trellis_core.dpo_branch import DPOBranchConfig, build_t_pairs, select_branch_windows


def _make_candidates(rng, radius, perturb_scale, protrusion_factor):
    """A reference icosphere + a delta with a small per-vertex perturbation
    and an exaggerated protrusion, so overhang/thickness histograms have
    something more interesting to show than a uniform sphere."""
    ref_mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    delta_verts = ref_mesh.vertices + rng.normal(scale=perturb_scale, size=ref_mesh.vertices.shape)
    delta_mesh = trimesh.Trimesh(vertices=delta_verts, faces=ref_mesh.faces, process=False)
    delta_mesh.vertices[0] *= protrusion_factor
    return ref_mesh, delta_mesh


def main():
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "synthetic-demo")
    w = TraceWriter(run_dir)
    num_branches = 2

    w.emit("session_start", {
        "image": "shoe_input.png", "pipeline_type": "512", "steps": 24, "seed": 42,
        "target_faces": 1000000, "model_id": "microsoft/TRELLIS.2-4B (synthetic demo)",
        "num_branches": num_branches, "vanilla_too": True, "pid": os.getpid(),
    })
    w.emit("stage", {"name": "load_pipeline", "model_id": "microsoft/TRELLIS.2-4B"})
    w.emit("stage", {"name": "pipeline_loaded", "seconds": 89.0})
    w.emit("stage", {"name": "preprocess_image"})
    w.emit("stage", {"name": "sample_sparse_structure"})
    w.emit("stage", {"name": "sparse_structure_done", "n_voxels": 1475})

    # The REAL schedule + REAL multi-branch window selection -- num_branches=2
    # spreads across [0.3, 0.7] (see DPOBranchConfig.effective_t_branches),
    # select_branch_windows keeps the two forks non-overlapping.
    t_pairs = build_t_pairs(24)
    w.emit("run_start", {"steps": 24, "rescale_t": 1.0, "t_pairs": t_pairs})
    cfg = DPOBranchConfig(num_branches=num_branches)
    windows = select_branch_windows(t_pairs, cfg.effective_t_branches(), k=2)
    assert len(windows) == num_branches, windows  # steps=24 easily fits 2 forks of k=2

    weights = gj.JudgeWeights(d_min=0.02, delta=0.05)  # dpo_branch.py's own default mid-generation weights
    rng = np.random.default_rng(0)

    # Branch 0: judge prefers delta, steering nudges it further in that
    # direction -- the exact loss/proxy/rms history captured from this
    # session's real dpo_branch.py smoke test on real Apple Silicon hardware.
    # Branch 1: judge prefers the reference instead, so the two branches in
    # this one demo trace show both possible outcomes.
    branch_specs = [
        {
            "radius": 0.4, "resumed_from": "delta",
            "loss_hist": [0.8544183373451233, 0.7590264678001404, 0.6645756363868713],
            "proxy_hist": [785.5155639648438, 785.68798828125, 785.8736572265625],
            "rms_hist": [0.020105773583054543, 0.02010669931769371, 0.020107971504330635],
            "proxy_reference": 785.815673828125,
        },
        {
            "radius": 0.36, "resumed_from": "reference",
            "loss_hist": [0.71203, 0.69881, 0.70452],
            "proxy_hist": [612.409, 611.882, 612.031],
            "rms_hist": [0.019883, 0.019901, 0.019897],
            "proxy_reference": 615.774,
        },
    ]

    cursor = 0
    for branch_idx, ((i_branch, k), spec) in enumerate(zip(windows, branch_specs)):
        for i, (t, t_prev) in enumerate(t_pairs[cursor:i_branch]):
            w.emit("sampler_step", {"phase": "pre_branch", "branch_index": branch_idx, "index": i, "t": t, "t_prev": t_prev})

        branch_t, resume_t = t_pairs[i_branch][0], t_pairs[i_branch + k - 1][1]
        w.emit("branch_point", {
            "branch_index": branch_idx, "n_branches": num_branches,
            "branch_t": branch_t, "resume_t": resume_t, "i_branch": i_branch, "k": k,
            "n_voxels": 1475, "out_of_window": False,
            "branch_noise_scale": 0.02, "trust_region": 0.02 * 3.0,
            "judge_weights": weights,
        })

        ref_mesh, delta_mesh = _make_candidates(rng, spec["radius"], 0.01, 1.4)
        score_ref, details_ref = gj.score_mesh_detailed(ref_mesh, weights, np.random.default_rng(1))
        score_delta, details_delta = gj.score_mesh_detailed(delta_mesh, weights, np.random.default_rng(1))
        w.emit("candidate_scored", {"branch_index": branch_idx, "which": "reference", "mesh": ref_mesh, "score": score_ref, "cmp": score_ref.total, "details": details_ref})
        w.emit("candidate_scored", {"branch_index": branch_idx, "which": "delta_initial", "mesh": delta_mesh, "score": score_delta, "cmp": score_delta.total, "details": details_delta})

        for i in range(3):
            w.emit("grad_step", {
                "branch_index": branch_idx, "step": i,
                "loss": spec["loss_hist"][i], "proxy": spec["proxy_hist"][i], "rms": spec["rms_hist"][i],
                "proxy_reference": spec["proxy_reference"],
            })

        final_verts = ref_mesh.vertices + rng.normal(scale=0.006, size=ref_mesh.vertices.shape)
        final_mesh = trimesh.Trimesh(vertices=final_verts, faces=ref_mesh.faces, process=False)
        final_mesh.vertices[0] *= 1.2
        score_final, details_final = gj.score_mesh_detailed(final_mesh, weights, np.random.default_rng(1))
        w.emit("candidate_scored", {
            "branch_index": branch_idx, "which": "delta_final", "mesh": final_mesh,
            "score": score_final, "cmp": score_final.total, "details": details_final,
        })

        w.emit("resume", {"branch_index": branch_idx, "resumed_from": spec["resumed_from"], "resume_t": resume_t})
        cursor = i_branch + k

        if branch_idx == 0:
            # keep the last branch's scores/meshes around for the closing print
            b0_scores = (score_ref, score_delta, score_final)

    for i, (t, t_prev) in enumerate(t_pairs[cursor:]):
        w.emit("sampler_step", {"phase": "post_branch", "branch_index": num_branches - 1, "index": i, "t": t, "t_prev": t_prev})

    w.emit("run_end", {"returned_report": True})  # dpo_branch's own module-level event (NOT terminal, see trace.py)

    w.emit("stage", {"name": "decode_shape_slat"})
    w.emit("stage", {"name": "print_prep", "raw_vertex_count": 498932, "raw_face_count": 999999})

    # The raw (pre-print-prep) final mesh -- dpo_generation.py emits this from
    # the actual decoded-and-cleaned-of-specks mesh, before
    # printable.process_object()'s watertight repair/decimation/solid-infill.
    raw_base = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    raw_verts = raw_base.vertices + rng.normal(scale=0.004, size=raw_base.vertices.shape)
    raw_mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_base.faces, process=False)
    w.emit("raw_result", {"mesh": raw_mesh})

    # Real output files under <run_dir>/output/, matching runner.py's actual
    # naming convention (output_prefix + ".glb"/".stl") -- so the "Downloads"
    # section's links are genuinely clickable in this demo, not just UI chrome
    # pointing at paths that don't exist.
    out_dir = os.path.join(run_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    printable_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    printable_mesh.export(os.path.join(out_dir, "model.glb"))
    printable_mesh.export(os.path.join(out_dir, "model.stl"))

    w.emit("printable_result", {
        "watertight": True,
        "diagnostics": {"overhang_pct": 8.4, "thin_wall_warnings": 12},
        "fidelity": {"chamfer": 0.0021, "hausdorff": 0.019, "volume_change_pct": 1.8},
        "vertex_count": 250133, "face_count": 500000,
        "generation_seconds": 1469.0, "postprocess_seconds": 38.2,
        "glb_path": "output/model.glb", "stl_path": "output/model.stl",
    })

    # Vanilla (non-DPO) comparison output -- run_generation() only ever
    # exports glb+obj (no stl), matching devlab/runner.py's --vanilla-too path.
    vanilla_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.42)
    vanilla_mesh.export(os.path.join(out_dir, "vanilla.glb"))
    vanilla_mesh.export(os.path.join(out_dir, "vanilla.obj"))
    w.emit("vanilla_result", {
        "vertex_count": len(vanilla_mesh.vertices), "face_count": len(vanilla_mesh.faces),
        "seconds": 1290.0, "glb_path": "output/vanilla.glb", "obj_path": "output/vanilla.obj",
    })

    w.emit("session_end", {"ok": True, "summary": {"note": "synthetic demo trace for UI verification (2 branches)"}})
    w.close()
    print(f"Wrote synthetic trace to {run_dir}")
    print(f"  windows={windows}")
    print(f"  branch 0: reference S={b0_scores[0].total:.4f}  delta_initial S={b0_scores[1].total:.4f}  delta_final S={b0_scores[2].total:.4f}")


if __name__ == "__main__":
    main()
