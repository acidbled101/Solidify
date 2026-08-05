"""
CPU-only unit tests for devlab/metrics.py.

Same self-contained-script convention as the rest of this repo:

    python devlab/metrics_test.py

metrics.py is what turns a pile of traces into the numbers a decision gets made
on, so the things worth testing are the ones that would silently corrupt that
decision rather than crash:

  * synthetic demo traces must be EXCLUDED from aggregates. Including them
    moves the mean steering gain from -0.005 to -0.030 -- a sign-preserving,
    magnitude-6x distortion that looks entirely plausible on a dashboard.
  * verdict_flipped must key off cmp_*, not score.total. When thickness is
    invalid the judge's actual comparison value differs from the reported
    total, and using the wrong one silently mis-labels which branch won.
  * gain decomposition must split "what the random draw bought" from "what the
    gradient added" in the right direction.
  * partial/truncated traces (a crashed run) must degrade to None, not raise.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devlab import metrics


def _write_trace(root: Path, run_id: str, events: list) -> Path:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "trace.jsonl", "w") as f:
        for i, (etype, payload) in enumerate(events):
            f.write(json.dumps({"seq": i, "t": 1000.0 + i, "type": etype, "payload": payload}) + "\n")
    return d


def _branch_events(branch=0, ref=0.30, d0=0.35, d1=0.40, resumed="delta", grad=True):
    """A complete, well-formed fork."""
    ev = [
        ("branch_point", {"branch_index": branch, "branch_t": 0.6, "resume_t": 0.5,
                          "i_branch": 5, "k": 2, "n_voxels": 2818, "out_of_window": False,
                          "branch_noise_scale": 0.1, "trust_region": 0.3}),
        ("branch_perturbation", {"branch_index": branch, "eps_rms": 0.1, "base_rms": 0.75, "relative": 0.133}),
        ("candidate_scored", {"branch_index": branch, "which": "reference",
                              "score": {"total": ref, "thickness_valid": True}, "cmp": ref}),
        ("candidate_scored", {"branch_index": branch, "which": "delta_initial",
                              "score": {"total": d0}, "cmp": d0}),
    ]
    if grad:
        for i, loss in enumerate([0.9, 0.8, 0.7]):
            ev.append(("grad_step", {"branch_index": branch, "step": i, "loss": loss,
                                     "proxy": 800.0 + i, "rms": 0.1, "proxy_reference": 795.0}))
        ev.append(("candidate_scored", {"branch_index": branch, "which": "delta_final",
                                        "score": {"total": d1}, "cmp": d1}))
    ev.append(("resume", {"branch_index": branch, "resumed_from": resumed}))
    return ev


def _full_run(**kw):
    return ([("session_start", {"image": "a.png", "pipeline_type": "512", "seed": 42,
                                "num_branches": 1, "model_id": "microsoft/TRELLIS.2-4B"}),
             ("run_start", {"steps": 12})]
            + _branch_events(**kw)
            + [("printable_result", {"watertight": False, "vertex_count": 10, "face_count": 20,
                                     "generation_seconds": 500.0}),
               ("session_end", {"ok": True})])


def test_synthetic_runs_are_excluded_from_aggregates():
    """The distortion this file mainly exists to prevent."""
    root = Path(tempfile.mkdtemp())
    try:
        # Two real forks with a small POSITIVE steering gain...
        _write_trace(root, "20260101-000000-aaaaaa", _full_run(ref=0.30, d0=0.35, d1=0.37))
        _write_trace(root, "20260101-000001-bbbbbb", _full_run(ref=0.20, d0=0.25, d1=0.27))
        # ...and one synthetic trace with a huge NEGATIVE one.
        _write_trace(root, "synthetic-demo", _full_run(ref=0.0, d0=0.50, d1=-1.0))

        data = metrics.build(root)
        s = data["summary"]
        assert s["n_synthetic_excluded"] == 1, s
        assert s["n_branches_scored"] == 2, f"synthetic fork leaked into aggregates: {s}"
        assert abs(s["mean_gain_steering"] - 0.02) < 1e-9, s["mean_gain_steering"]
        # but it must still be present, labelled, in the rows
        rows = data["branch_rows"]
        assert len(rows) == 3, rows
        assert sum(1 for r in rows if r["is_synthetic"]) == 1
        print(f"  synthetic excluded from aggregates (mean {s['mean_gain_steering']:+.3f}) "
              f"but retained + labelled in {len(rows)} CSV rows")
    finally:
        shutil.rmtree(root)


def test_verdict_flip_and_gain_decomposition():
    """flip detection and the random-vs-gradient split, in both directions."""
    root = Path(tempfile.mkdtemp())
    try:
        # delta loses initially (0.25 < 0.30) then WINS after steering (0.40) -> flip
        _write_trace(root, "20260101-000000-flip01", _full_run(ref=0.30, d0=0.25, d1=0.40))
        # delta wins initially and still wins -> no flip
        _write_trace(root, "20260101-000001-same01", _full_run(ref=0.30, d0=0.35, d1=0.38))
        # delta wins initially then LOSES after steering -> flip the other way
        _write_trace(root, "20260101-000002-flip02", _full_run(ref=0.30, d0=0.35, d1=0.20))

        rows = {r["run_id"][-6:]: r for r in metrics.build(root)["branch_rows"]}

        f1 = rows["flip01"]
        assert f1["delta_won_initial"] is False and f1["delta_won_final"] is True
        assert f1["verdict_flipped"] is True
        assert abs(f1["gain_perturbation"] - (-0.05)) < 1e-9, f1["gain_perturbation"]
        assert abs(f1["gain_steering"] - 0.15) < 1e-9, f1["gain_steering"]
        assert abs(f1["gain_total"] - 0.10) < 1e-9, f1["gain_total"]

        assert rows["same01"]["verdict_flipped"] is False
        f2 = rows["flip02"]
        assert f2["delta_won_initial"] is True and f2["delta_won_final"] is False
        assert f2["verdict_flipped"] is True
        assert f2["gain_steering"] < 0

        s = metrics.build(root)["summary"]
        assert s["n_verdict_flipped"] == 2 and s["flip_rate"] == 0.6667, s
        print("  flip detected in both directions; gain split into "
              f"random {f1['gain_perturbation']:+.2f} / gradient {f1['gain_steering']:+.2f}")
    finally:
        shutil.rmtree(root)


def test_partial_and_broken_traces_degrade_quietly():
    """A crashed run must produce a row with Nones, never an exception.

    This is not hypothetical: 5 of the traces on disk are errored runs that
    stopped between candidate_scored and grad_step.
    """
    root = Path(tempfile.mkdtemp())
    try:
        # errored mid-fork: reference + delta_initial scored, no steering, no end
        partial = ([("session_start", {"image": "a.png", "seed": 1})]
                   + _branch_events(d1=None, grad=False)
                   + [("error", {"message": "MPS backend out of memory"})])
        _write_trace(root, "20260101-000000-part01", partial)
        _write_trace(root, "20260101-000001-empty1", [])
        (root / "not-a-run").mkdir()  # directory with no trace.jsonl
        _write_trace(root, "20260101-000002-good01", _full_run())

        data = metrics.build(root)
        by = {r["run_id"][-6:]: r for r in data["runs"]}
        assert by["part01"]["status"] == "error"
        assert "out of memory" in (by["part01"]["error"] or "")
        assert by["empty1"]["status"] == "empty"
        assert not any(r["run_id"] == "not-a-run" for r in data["runs"])

        prow = [r for r in data["branch_rows"] if r["run_id"].endswith("part01")][0]
        assert prow["score_delta_final"] is None
        assert prow["gain_steering"] is None
        assert prow["verdict_flipped"] is None, "must not guess a verdict it cannot know"
        assert prow["delta_won_initial"] is True  # 0.35 > 0.30, knowable

        # the incomplete fork must not contaminate the aggregate
        assert data["summary"]["n_branches_scored"] == 1
        print("  errored/empty/non-run directories degrade to None without raising; "
              "unknowable verdicts stay None rather than defaulting to False")
    finally:
        shutil.rmtree(root)


def test_writes_all_three_files_and_csv_roundtrips():
    root = Path(tempfile.mkdtemp())
    try:
        _write_trace(root, "20260101-000000-aaaaaa", _full_run())
        paths = metrics.write(metrics.build(root), root)
        names = {p.name for p in paths}
        assert names == {"metrics.json", "metrics_runs.csv", "metrics_branches.csv"}, names
        for p in paths:
            assert p.exists() and p.stat().st_size > 0, p

        import csv
        with open(root / "metrics_branches.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert set(rows[0]).issuperset({"gain_steering", "verdict_flipped", "is_synthetic"})
        assert abs(float(rows[0]["gain_steering"]) - 0.05) < 1e-9
        with open(root / "metrics.json") as f:
            assert "summary" in json.load(f)
        print(f"  wrote {len(paths)} files; branches.csv round-trips with "
              f"{len(rows[0])} columns")
    finally:
        shutil.rmtree(root)


TESTS = [
    test_synthetic_runs_are_excluded_from_aggregates,
    test_verdict_flip_and_gain_decomposition,
    test_partial_and_broken_traces_degrade_quietly,
    test_writes_all_three_files_and_csv_roundtrips,
]


def main():
    failures = []
    for fn in TESTS:
        print(f"{fn.__name__}:")
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"  FAILED: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} FAILED")
        return 1
    print(f"All {len(TESTS)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
