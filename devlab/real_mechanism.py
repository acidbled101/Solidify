"""Panels M1-M4's data, rebuilt from REAL TRELLIS.2 runs instead of the toy.

Companion to devlab/alignment_experiment.py, which trains a 33.8K-parameter
TinyFlow on a 2-parameter synthetic frustum to answer the mechanism question
at n=3000 -- necessary because the real pipeline cannot supply that many
samples. This module answers the SAME four questions (does the field's
velocity differ under steering, do the sampled paths straighten, does the
reward distribution shift, what happens over gradient steps) using only real
data already on disk: no synthetic shape, no trained toy, no new GPU time.

What real data supports and what it doesn't
---------------------------------------------
The shipped pipeline has ONE frozen flow model (frozen_parameters() in
dpo_branch.py asserts this) and does inference-time LATENT steering, not
weight training. There is no "aligned model" to compare against a "base
model" -- there is one v_theta, evaluated at different inputs. So this module
does NOT attempt a toy-shaped "base vs aligned field" comparison (that
question has no real referent); instead it compares how v_theta behaves along
the REFERENCE (unperturbed) continuation vs the DELTA (perturbed, then
steered) continuation of the SAME branch -- the real analogue of "does
steering change the field's behavior", using the same trajectories the
pipeline actually explored.

Two independent data sources, with different n:
  * grad_step / candidate_scored events exist for every scored branch (any
    run past the point where multi-branch scoring landed). Used for
    reward_shift and grad_step_history.
  * branch_latent events (LatentProbe) exist only for runs after that
    instrumentation was added. Used for curvature and velocity_signature,
    and n is smaller as a result -- reported explicitly per section.

Cross-run pooling caveat: LatentProbe's projection basis is reseeded whenever
a run's voxel count differs (see LatentProbe._ensure_basis), and voxel count
varies run to run (2818 vs 3328 measured across the 4 runs with telemetry).
So raw 2D projected COORDINATES from different runs are not on the same axes
and must never be overlaid on one chart. What IS comparable across runs:
scalar quantities computed from the full (non-projected) feature tensor --
rms, v_rms -- and scale-free path shape statistics (tortuosity, normalised
curvature), which do not depend on which basis produced the points. Both are
used here; raw cross-run positions are not.

Run:  python devlab/real_mechanism.py --print
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from devlab import metrics
from devlab.trace import _histogram
from trellis_core import flow_dpo

TRACES_DIR = Path(__file__).resolve().parent / "traces"


# ---------------------------------------------------------------------------
# Segment trajectories from a branch's latent_path (LatentProbe events)
# ---------------------------------------------------------------------------


def _segment_points(latent_path: dict, which: str) -> List[List[float]]:
    """The fork origin followed by `which`'s continuation-step projections.

    Prepending the fork origin is not an approximation -- the reference and
    delta continuations genuinely start there, it's just recorded as a
    separate event (branch_latent/which=fork) rather than index 0 of each
    segment. Including it turns a k=2-step segment (the pipeline default)
    into 3 points, which is the minimum path_curvature needs for one real
    second-difference rather than zero.
    """
    fork = latent_path.get("fork") or []
    pts: List[List[float]] = []
    if fork and fork[0].get("proj"):
        pts.append(fork[0]["proj"])
    for row in sorted(latent_path.get(which) or [], key=lambda r: r.get("index", 0)):
        if row.get("proj"):
            pts.append(row["proj"])
    return pts


def _segment_v_rms(latent_path: dict, which: str) -> Optional[float]:
    """Mean v_rms across `which`'s continuation steps (basis-independent)."""
    vals = [row["v_rms"] for row in (latent_path.get(which) or []) if row.get("v_rms") is not None]
    return float(np.mean(vals)) if vals else None


def _paired_bootstrap(diffs: List[float], n_boot: int = 4000, seed: int = 0) -> dict:
    if len(diffs) < 2:
        return {"mean": (float(diffs[0]) if diffs else None), "ci95": [None, None], "n": len(diffs)}
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(arr.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": len(diffs),
    }


# ---------------------------------------------------------------------------
# The four sections
# ---------------------------------------------------------------------------


def _reward_shift(branch_rows: List[dict]) -> dict:
    """Real judge scores, pooled: reference / random-perturbation / steered."""
    ref = [r["score_reference"] for r in branch_rows if r["score_reference"] is not None]
    rand = [r["score_delta_initial"] for r in branch_rows if r["score_delta_initial"] is not None]
    steer = [r["score_delta_final"] for r in branch_rows if r["score_delta_final"] is not None]
    diff_rand = [r["score_delta_initial"] - r["score_reference"] for r in branch_rows
                 if r["score_delta_initial"] is not None and r["score_reference"] is not None]
    diff_steer = [r["score_delta_final"] - r["score_reference"] for r in branch_rows
                  if r["score_delta_final"] is not None and r["score_reference"] is not None]
    return {
        "reference": _histogram(ref, bins=12),
        "delta_initial": _histogram(rand, bins=12),
        "delta_final": _histogram(steer, bins=12),
        "reference_mean": float(np.mean(ref)) if ref else None,
        "delta_initial_mean": float(np.mean(rand)) if rand else None,
        "delta_final_mean": float(np.mean(steer)) if steer else None,
        "delta_initial_vs_reference": _paired_bootstrap(diff_rand, seed=1),
        "delta_final_vs_reference": _paired_bootstrap(diff_steer, seed=2),
        "n": len(ref),
    }


def _curvature_and_velocity(runs: List[dict]) -> dict:
    """Path shape (scale-free) and velocity magnitude, per real branch with
    latent telemetry. Both quantities survive cross-run pooling; see module
    docstring for why raw projected positions do not."""
    per_branch = []
    for run in runs:
        for b in run.get("branches", []):
            lat = b.get("latent_path") or {}
            if not lat:
                continue
            row = {"run_id": run["run_id"], "branch_index": b["branch_index"]}
            any_seg = False
            for which in ("reference", "delta_initial", "delta_final"):
                pts = _segment_points(lat, which)
                if len(pts) >= 3:
                    traj = torch.tensor(pts, dtype=torch.float32).unsqueeze(1)  # (steps, 1, 2)
                    row[f"tortuosity_{which}"] = float(flow_dpo.path_tortuosity(traj)[0])
                    row[f"curvature_norm_{which}"] = float(flow_dpo.path_curvature_normalized(traj)[0])
                    any_seg = True
                v_rms = _segment_v_rms(lat, which)
                if v_rms is not None:
                    row[f"v_rms_{which}"] = v_rms
                    any_seg = True
            if any_seg:
                per_branch.append(row)

    def _pool(key):
        return [r[key] for r in per_branch if key in r]

    tort = {w: _pool(f"tortuosity_{w}") for w in ("reference", "delta_initial", "delta_final")}
    means = {f"tortuosity_{w}_mean": (float(np.mean(v)) if v else None) for w, v in tort.items()}
    return {
        "per_branch": per_branch,
        "n_branches": len(per_branch),
        "tortuosity_reference": _histogram(tort["reference"], bins=8),
        "tortuosity_delta_initial": _histogram(tort["delta_initial"], bins=8),
        "tortuosity_delta_final": _histogram(tort["delta_final"], bins=8),
        **means,
    }


def _grad_step_history(branch_rows_full: List[dict], runs: List[dict]) -> dict:
    """proxy-gap and trust-region-normalised rms, pooled by step INDEX across
    every real branch that reached that step.

    NOTHING here is a pooled mean across branches, and that is a deliberate
    correction rather than a style choice. branch_noise_scale ranges 0.02-1.0
    across the branches on disk (a 50x spread -- several runs this session
    used larger noise to make effects visible during instrumentation work, so
    this is not a controlled sweep). Measured consequence: proxy_gap at step 0
    ranges from -0.06 (noise=0.02) to +1563 (noise=1.0). An earlier draft of
    this function pooled proxy_gap as a per-step mean and produced a dramatic
    "drop" from 109 to 2.7 between step 2 and step 3 -- which was purely the
    large-noise branches dropping out of the cohort at step 3, not any trend.
    Raw DPO loss has the same problem (0.001 to 1562 at step 0).

    So each branch is emitted as its OWN trace, normalised to its own step-0
    proxy gap, making the SHAPE of each comparable while never averaging
    incommensurable magnitudes. n per step is still reported so cohort
    shrinkage stays visible.
    """
    by_run = {r["run_id"]: r for r in runs}
    per_branch: List[dict] = []
    counts: Dict[int, int] = {}
    rms_fracs: List[float] = []
    longest = None  # (run_id, branch_index, length) -- the deepest real case study

    for row in branch_rows_full:
        run = by_run.get(row["run_id"])
        if not run:
            continue
        b = next((x for x in run["branches"] if x["branch_index"] == row["branch_index"]), None)
        if not b or not b.get("loss_history"):
            continue
        proxy_ref, trust_region = b.get("proxy_reference"), b.get("trust_region")
        if longest is None or len(b["loss_history"]) > longest[2]:
            longest = (row["run_id"], row["branch_index"], len(b["loss_history"]))

        gaps = ([p - proxy_ref for p in b["proxy_history"]] if proxy_ref is not None else [])
        base = gaps[0] if gaps and abs(gaps[0]) > 1e-9 else None
        per_branch.append({
            "run_id": row["run_id"], "branch_index": row["branch_index"],
            "branch_noise_scale": b.get("branch_noise_scale"),
            "n_steps": len(b["loss_history"]),
            "proxy_gap": gaps,
            # Each branch relative to its own starting gap: 1.0 at step 0 by
            # construction, so the curves show direction and rate of change
            # rather than magnitudes that differ 25000x between configs.
            "proxy_gap_relative": ([g / base for g in gaps] if base else None),
            "rms_frac": ([r / trust_region for r in b["rms_history"]] if trust_region else None),
        })
        for i in range(len(b["loss_history"])):
            counts[i] = counts.get(i, 0) + 1
        if trust_region:
            rms_fracs.extend(r / trust_region for r in b["rms_history"])

    steps = [{"index": i, "n_branches": counts[i]} for i in sorted(counts)]

    case_study = None
    if longest and longest[2] > 5:  # meaningfully deeper than the pipeline default of 3
        run = by_run[longest[0]]
        b = next(x for x in run["branches"] if x["branch_index"] == longest[1])
        case_study = {
            "run_id": longest[0], "branch_index": longest[1], "n_steps": longest[2],
            "loss": b["loss_history"], "proxy": b["proxy_history"], "rms": b["rms_history"],
            "proxy_reference": b.get("proxy_reference"), "trust_region": b.get("trust_region"),
            "branch_noise_scale": b.get("branch_noise_scale"),
        }

    # The trust region turns out never to bind. delta's RMS sits at ~1/3 of
    # its cap on every branch at every step, because trust_region is
    # branch_noise_scale * delta_max_norm_ratio (=3.0) while the steered delta
    # stays at essentially its initial random magnitude -- i.e. the gradient
    # steps rotate delta far more than they grow it. Reported as a scalar
    # finding rather than a chart: a flat line at 1/3 is not a time series.
    rms_summary = None
    if rms_fracs:
        arr = np.asarray(rms_fracs)
        rms_summary = {
            "mean_frac_of_trust_region": float(arr.mean()),
            "min": float(arr.min()), "max": float(arr.max()),
            "n_observations": int(arr.size),
            "trust_region_ever_binding": bool(arr.max() > 0.99),
        }
    return {"steps": steps, "per_branch": per_branch,
            "rms_summary": rms_summary, "case_study": case_study}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(traces_dir: Path = TRACES_DIR) -> dict:
    base = metrics.build(traces_dir)
    runs = [r for r in base["runs"] if not r.get("is_synthetic")]
    branch_rows = [b for b in base["branch_rows"] if not b["is_synthetic"]]
    n_with_latent = sum(1 for r in runs for b in r.get("branches", []) if b.get("latent_path"))

    return {
        "meta": {
            "generated_by": "devlab/real_mechanism.py",
            "n_runs_total": len(runs),
            "n_runs_ok": sum(1 for r in runs if r["status"] == "ok"),
            "n_branches_scored": sum(1 for b in branch_rows if b["score_reference"] is not None),
            "n_branches_with_latent": n_with_latent,
            "note": (
                "Every number below is read from real trace.jsonl files produced by "
                "actual TRELLIS.2-4B generations (devlab/runner.py). No synthetic shape, "
                "no trained toy model, no new GPU time. n_branches_with_latent is smaller "
                "than n_branches_scored because latent/velocity telemetry (LatentProbe) "
                "was added partway through this session -- only runs since then carry it."
            ),
        },
        "reward_shift": _reward_shift(branch_rows),
        "curvature": _curvature_and_velocity(runs),
        "grad_step_history": _grad_step_history(branch_rows, runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(TRACES_DIR))
    ap.add_argument("--print", action="store_true", dest="do_print")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = build(Path(args.traces))
    m = data["meta"]
    print(f"{m['n_runs_ok']}/{m['n_runs_total']} real runs ok, "
          f"{m['n_branches_scored']} branches scored, "
          f"{m['n_branches_with_latent']} with latent telemetry")
    rs = data["reward_shift"]
    print(f"  reward: ref={rs['reference_mean']:.4f} rand={rs['delta_initial_mean']:.4f} "
          f"steer={rs['delta_final_mean']:.4f}  (n={rs['n']})")
    print(f"  steer-vs-ref: mean={rs['delta_final_vs_reference']['mean']:.4f} "
          f"CI={rs['delta_final_vs_reference']['ci95']}")
    cv = data["curvature"]
    print(f"  curvature: n_branches={cv['n_branches']}  "
          f"tortuosity ref={cv.get('tortuosity_reference_mean')} "
          f"rand={cv.get('tortuosity_delta_initial_mean')} steer={cv.get('tortuosity_delta_final_mean')}")
    gh = data["grad_step_history"]
    rs2 = gh["rms_summary"]
    print(f"  grad steps: {len(gh['per_branch'])} branch traces, "
          f"n at step0={gh['steps'][0]['n_branches'] if gh['steps'] else 0}, "
          f"deepest={gh['case_study']['n_steps'] if gh['case_study'] else 0} steps "
          f"({gh['case_study']['run_id'] if gh['case_study'] else 'none'})")
    if rs2:
        print(f"  delta RMS sits at {rs2['mean_frac_of_trust_region']:.4f} of its trust region "
              f"(range {rs2['min']:.4f}-{rs2['max']:.4f}, n={rs2['n_observations']}); "
              f"ever binding: {rs2['trust_region_ever_binding']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(data, f)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
