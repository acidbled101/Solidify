"""Aggregate every run's metrics into files you can analyse later.

Each run's trace.jsonl is append-only and self-describing, which is great for
replay and useless for "did steering ever actually help, across all runs".
This walks every trace and flattens it into three artifacts under
devlab/traces/:

    metrics.json          full nested records, one per run
    metrics_runs.csv      one row per run      -- run-level comparison
    metrics_branches.csv  one row per BRANCH   -- the analysis unit that matters

The branch-level CSV is the important one. A multi-branch run forks several
times and each fork is an independent trial of the same question, so pooling
them is what turns n=1 runs into n=(runs x branches) observations.

THE COLUMN THIS FILE EXISTS FOR
-------------------------------
`verdict_flipped`: did the gradient steering change which branch the judge
preferred? dpo_branch records both `delta_branch_won_initial` (judge's verdict
on the RANDOM perturbation) and `delta_branch_won_final` (verdict after the
gradient steps). If those two agree on every branch across every run, the
gradient is doing nothing the random draw did not already do, and best-of-N
sampling dominates the whole approach at lower cost.

That comparison is the cheapest decisive experiment available on this pipeline
and it needs no new GPU time -- only aggregation over runs already on disk.

Run:  python devlab/metrics.py            (writes the three files)
      python devlab/metrics.py --print    (also dumps a summary table)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devlab.trace import TraceReader

TRACES_DIR = Path(__file__).resolve().parent / "traces"


def _get(d: Optional[dict], *keys, default=None):
    """Nested lookup that tolerates missing intermediate dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _first_last(seq) -> tuple:
    if not seq:
        return None, None
    return seq[0], seq[-1]


def collect_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Flatten one run's trace into a record. Returns None if unreadable."""
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        return None
    try:
        events = TraceReader(run_dir).read_all()
    except Exception as e:
        return {"run_id": run_dir.name, "status": "unreadable", "error": str(e)}
    if not events:
        return {"run_id": run_dir.name, "status": "empty"}

    by_type: Dict[str, List[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("type", "?"), []).append(e)

    start = _get(by_type, "session_start", default=[{}])[0].get("payload", {}) if by_type.get("session_start") else {}
    end = by_type.get("session_end", [{}])[-1].get("payload", {}) if by_type.get("session_end") else {}
    err = by_type.get("error", [{}])[-1].get("payload", {}) if by_type.get("error") else {}
    printable = by_type.get("printable_result", [{}])[-1].get("payload", {}) if by_type.get("printable_result") else {}
    vanilla = by_type.get("vanilla_result", [{}])[-1].get("payload", {}) if by_type.get("vanilla_result") else {}
    run_start = by_type.get("run_start", [{}])[0].get("payload", {}) if by_type.get("run_start") else {}

    t0 = events[0].get("t")
    t1 = events[-1].get("t")
    status = ("ok" if end.get("ok") else
              "error" if err else
              "incomplete")

    # ---- per-branch records -------------------------------------------------
    branches: Dict[int, Dict[str, Any]] = {}

    def slot(i: int) -> Dict[str, Any]:
        return branches.setdefault(int(i), {
            "branch_index": int(i), "scores": {}, "cmp": {},
            "loss_history": [], "proxy_history": [], "rms_history": [],
            "n_grad_steps": 0, "latent_path": {},
        })

    for e in by_type.get("branch_point", []):
        p = e["payload"]
        b = slot(p.get("branch_index", 0))
        b.update({
            "branch_t": p.get("branch_t"), "resume_t": p.get("resume_t"),
            "i_branch": p.get("i_branch"), "k": p.get("k"),
            "n_voxels": p.get("n_voxels"), "out_of_window": p.get("out_of_window"),
            "branch_noise_scale": p.get("branch_noise_scale"),
            "trust_region": p.get("trust_region"),
        })
    for e in by_type.get("branch_perturbation", []):
        p = e["payload"]
        b = slot(p.get("branch_index", 0))
        b.update({"eps_rms": p.get("eps_rms"), "base_rms": p.get("base_rms"),
                  "perturbation_relative": p.get("relative")})
    for e in by_type.get("candidate_scored", []):
        p = e["payload"]
        b = slot(p.get("branch_index", 0))
        which = p.get("which")
        b["scores"][which] = _get(p, "score", "total")
        b["cmp"][which] = p.get("cmp")
        if which == "reference":
            b["thickness_valid_ref"] = _get(p, "score", "thickness_valid")
            b["watertight_ref"] = _get(p, "mesh", "watertight")
        b.setdefault("breakdown", {})[which] = {
            k: _get(p, "score", k) for k in
            ("detail_reward", "overhang_penalty", "thickness_penalty", "topology_penalty")
        }
    for e in by_type.get("grad_step", []):
        p = e["payload"]
        b = slot(p.get("branch_index", 0))
        b["loss_history"].append(p.get("loss"))
        b["proxy_history"].append(p.get("proxy"))
        b["rms_history"].append(p.get("rms"))
        b["proxy_reference"] = p.get("proxy_reference")
        b["n_grad_steps"] += 1
    for e in by_type.get("resume", []):
        p = e["payload"]
        slot(p.get("branch_index", 0))["resumed_from"] = p.get("resumed_from")
    for e in by_type.get("branch_latent", []):
        p = e["payload"]
        b = slot(p.get("branch_index", 0))
        b["latent_path"].setdefault(p.get("which"), []).append({
            "index": p.get("index"), "t": p.get("t"), **(p.get("latent") or {})
        })

    for b in branches.values():
        s = b["scores"]
        ref, d0, d1 = s.get("reference"), s.get("delta_initial"), s.get("delta_final")
        # The judge's verdict before and after steering. cmp_* is what actually
        # decides the winner (it can differ from .total when thickness is
        # invalid), so prefer it and fall back to totals.
        c = b["cmp"]
        cref, cd0, cd1 = c.get("reference"), c.get("delta_initial"), c.get("delta_final")
        won_initial = (None if cref is None or cd0 is None else bool(cd0 > cref))
        won_final = (None if cref is None or cd1 is None else bool(cd1 > cref))
        b["delta_won_initial"] = won_initial
        b["delta_won_final"] = won_final
        b["verdict_flipped"] = (None if won_initial is None or won_final is None
                                else won_initial != won_final)
        # Decompose the total change into "what the random draw bought" vs
        # "what the gradient bought on top of it" -- the whole open question.
        b["gain_perturbation"] = (None if ref is None or d0 is None else d0 - ref)
        b["gain_steering"] = (None if d0 is None or d1 is None else d1 - d0)
        b["gain_total"] = (None if ref is None or d1 is None else d1 - ref)
        b["loss_first"], b["loss_last"] = _first_last(b["loss_history"])
        b["rms_first"], b["rms_last"] = _first_last(b["rms_history"])
        b["proxy_first"], b["proxy_last"] = _first_last(b["proxy_history"])
        b["latent_steps_traced"] = sum(len(v) for v in b["latent_path"].values())

    sampler_steps = by_type.get("sampler_step", [])
    has_latent = any(e["payload"].get("latent") for e in sampler_steps)

    is_synthetic = (
        "synthetic" in run_dir.name.lower()
        or "synthetic" in str(start.get("model_id", "")).lower()
        or "synthetic" in str(_get(end, "summary", "note", default="")).lower()
    )

    return {
        "run_id": run_dir.name,
        "status": status,
        "is_synthetic": is_synthetic,
        "error": err.get("message") or err.get("error") or (str(err) if err else None),
        "started_at": t0,
        "duration_seconds": (round(t1 - t0, 1) if t0 and t1 else None),
        "image": start.get("image"),
        "pipeline_type": start.get("pipeline_type"),
        "steps_requested": start.get("steps"),
        "steps_actual": run_start.get("steps"),
        "seed": start.get("seed"),
        "target_faces": start.get("target_faces"),
        "num_branches_requested": start.get("num_branches"),
        "num_branches_actual": len(branches),
        "model_id": start.get("model_id"),
        "pid": start.get("pid"),
        "n_events": len(events),
        "has_latent_telemetry": has_latent,
        "watertight": printable.get("watertight"),
        "vertex_count": printable.get("vertex_count"),
        "face_count": printable.get("face_count"),
        "generation_seconds": printable.get("generation_seconds"),
        "postprocess_seconds": printable.get("postprocess_seconds"),
        "overhang_pct": _get(printable, "diagnostics", "overhang_pct"),
        "chamfer": _get(printable, "fidelity", "chamfer"),
        "hausdorff": _get(printable, "fidelity", "hausdorff"),
        "volume_change_pct": _get(printable, "fidelity", "volume_change_pct"),
        "vanilla_seconds": vanilla.get("seconds"),
        "vanilla_face_count": vanilla.get("face_count"),
        "branches": [branches[k] for k in sorted(branches)],
    }


RUN_COLUMNS = [
    "run_id", "status", "is_synthetic", "started_at", "duration_seconds", "image", "pipeline_type",
    "steps_actual", "seed", "target_faces", "num_branches_requested", "num_branches_actual",
    "model_id", "n_events", "has_latent_telemetry", "watertight", "vertex_count",
    "face_count", "generation_seconds", "postprocess_seconds", "overhang_pct",
    "chamfer", "hausdorff", "volume_change_pct", "vanilla_seconds", "error",
]

BRANCH_COLUMNS = [
    "run_id", "status", "is_synthetic", "branch_index", "branch_t", "resume_t", "i_branch", "k",
    "n_voxels", "out_of_window", "branch_noise_scale", "trust_region",
    "eps_rms", "base_rms", "perturbation_relative",
    "score_reference", "score_delta_initial", "score_delta_final",
    "cmp_reference", "cmp_delta_initial", "cmp_delta_final",
    "delta_won_initial", "delta_won_final", "verdict_flipped",
    "gain_perturbation", "gain_steering", "gain_total",
    "n_grad_steps", "loss_first", "loss_last", "rms_first", "rms_last",
    "proxy_reference", "proxy_first", "proxy_last", "resumed_from",
    "latent_steps_traced",
]


def flatten_branch(run: dict, b: dict) -> dict:
    s, c = b.get("scores", {}), b.get("cmp", {})
    row = {k: b.get(k) for k in BRANCH_COLUMNS if k in b}
    row.update({
        "run_id": run["run_id"], "status": run["status"],
        "is_synthetic": run.get("is_synthetic", False),
        "score_reference": s.get("reference"),
        "score_delta_initial": s.get("delta_initial"),
        "score_delta_final": s.get("delta_final"),
        "cmp_reference": c.get("reference"),
        "cmp_delta_initial": c.get("delta_initial"),
        "cmp_delta_final": c.get("delta_final"),
    })
    return {k: row.get(k) for k in BRANCH_COLUMNS}


def build(traces_dir: Path = TRACES_DIR) -> dict:
    runs = []
    for d in sorted(traces_dir.iterdir()):
        if not d.is_dir():
            continue
        rec = collect_run(d)
        if rec:
            runs.append(rec)

    branch_rows = [flatten_branch(r, b) for r in runs for b in r.get("branches", [])]

    # Cross-run summary: the questions the CSVs exist to answer.
    # SYNTHETIC RUNS ARE EXCLUDED. devlab/traces/synthetic-demo exists to
    # exercise the UI without GPU time; its scores come from trimesh primitives
    # and its gradient history is replayed from an old smoke test. Pooling it
    # with real runs would corrupt every statistic here -- it alone moves the
    # mean steering gain from -0.005 to -0.030. It stays in the CSVs (labelled)
    # so nothing is hidden, and out of the aggregates so nothing is distorted.
    real = [b for b in branch_rows if not b.get("is_synthetic")]
    scored = [b for b in real if b["verdict_flipped"] is not None]
    steer = [b["gain_steering"] for b in real if b["gain_steering"] is not None]
    pert = [b["gain_perturbation"] for b in real if b["gain_perturbation"] is not None]
    def _mean(v):
        return (sum(v) / len(v)) if v else None

    def _stderr(v):
        if len(v) < 2:
            return None
        m = _mean(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) ** 0.5

    se = _stderr(steer)
    m_steer = _mean(steer)
    summary = {
        "n_runs": len(runs),
        "n_runs_ok": sum(1 for r in runs if r["status"] == "ok" and not r.get("is_synthetic")),
        "n_synthetic_excluded": sum(1 for r in runs if r.get("is_synthetic")),
        "n_branches": len(real),
        "n_branches_scored": len(scored),
        "n_verdict_flipped": sum(1 for b in scored if b["verdict_flipped"]),
        "flip_rate": (round(sum(1 for b in scored if b["verdict_flipped"]) / len(scored), 4)
                      if scored else None),
        "mean_gain_perturbation": (round(_mean(pert), 6) if pert else None),
        "mean_gain_steering": (round(m_steer, 6) if steer else None),
        "stderr_gain_steering": (round(se, 6) if se else None),
        "steering_significant": (None if se in (None, 0) or m_steer is None
                                 else bool(abs(m_steer) > 2 * se)),
        "n_steering_positive": sum(1 for g in steer if g > 0),
        "n_steering_negative": sum(1 for g in steer if g < 0),
        "interpretation": (
            "flip_rate is the headline: the fraction of forks where gradient steering "
            "CHANGED the judge's preferred branch. If it stays ~0 as n grows, the "
            "gradient is not contributing beyond the random perturbation and "
            "best-of-N sampling would dominate at lower cost. mean_gain_steering "
            "separates what the gradient bought from what the random draw bought "
            "(mean_gain_perturbation). Compare mean_gain_steering against "
            "stderr_gain_steering before reading anything into its sign."
        ),
    }
    return {"summary": summary, "runs": runs, "branch_rows": branch_rows}


def write(data: dict, traces_dir: Path = TRACES_DIR) -> List[Path]:
    out = []
    p = traces_dir / "metrics.json"
    with open(p, "w") as f:
        json.dump({"summary": data["summary"], "runs": data["runs"]}, f, indent=1)
    out.append(p)

    p = traces_dir / "metrics_runs.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in data["runs"]:
            w.writerow(r)
    out.append(p)

    p = traces_dir / "metrics_branches.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BRANCH_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in data["branch_rows"]:
            w.writerow(row)
    out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(TRACES_DIR))
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    data = build(Path(args.traces))
    paths = write(data, Path(args.traces))
    s = data["summary"]
    print(f"{s['n_runs']} runs ({s['n_runs_ok']} ok), {s['n_branches']} branches "
          f"({s['n_branches_scored']} scored)")
    print(f"  verdict flipped by steering: {s['n_verdict_flipped']}/{s['n_branches_scored']}"
          f"  (flip rate {s['flip_rate']})")
    print(f"  mean gain from random perturbation: {s['mean_gain_perturbation']}")
    print(f"  mean gain from gradient steering:   {s['mean_gain_steering']}"
          f" +/- {s['stderr_gain_steering']} (SE)"
          f"   (+{s['n_steering_positive']} / -{s['n_steering_negative']})"
          f"   significant={s['steering_significant']}")
    if s["n_synthetic_excluded"]:
        print(f"  ({s['n_synthetic_excluded']} synthetic run(s) excluded from aggregates)")
    for p in paths:
        print(f"  wrote {p}  ({os.path.getsize(p)/1024:.1f} KB)")

    if args.do_print:
        print()
        hdr = f"{'run':<26}{'st':<6}{'br':>3}{'S_ref':>9}{'S_d0':>9}{'S_d1':>9}{'steer':>9}{'flip':>6}"
        print(hdr)
        print("-" * len(hdr))
        for row in data["branch_rows"]:
            f = lambda v, w=9, p=4: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
            print(f"{row['run_id']:<26}{str(row['status'])[:5]:<6}{row['branch_index']:>3}"
                  f"{f(row['score_reference'])}{f(row['score_delta_initial'])}"
                  f"{f(row['score_delta_final'])}{f(row['gain_steering'])}"
                  f"{str(row['verdict_flipped']):>6}")


if __name__ == "__main__":
    main()
