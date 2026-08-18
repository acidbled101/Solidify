"""
CPU-only unit tests for experiments/dpo_inference_steering/inspector/trace.py. Matches the plain-assert +
__main__-runner style of trellis_core/geometric_judge_test.py /
dpo_branch_test.py (this repo has no pytest suite).

    python experiments/dpo_inference_steering/inspector/trace_test.py
"""
import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from experiments.dpo_inference_steering.inspector.trace import TraceReader, TraceWriter, _histogram


def _tmpdir():
    d = tempfile.mkdtemp(prefix="dpo_inspector_trace_test_")
    return d


def test_roundtrip_scalars():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("run_start", {"steps": 12, "t_branch": 0.5, "ok": True, "note": None})
        w.emit("grad_step", {"step": 0, "loss": 0.854, "proxy": 785.5156, "rms": 0.0201058})
        w.close()

        events = TraceReader(d).read_all()
        assert len(events) == 2
        assert events[0]["type"] == "run_start"
        assert events[0]["payload"]["steps"] == 12
        assert events[0]["payload"]["ok"] is True
        assert events[0]["payload"]["note"] is None
        assert events[1]["payload"]["loss"] == 0.854
        assert events[0]["seq"] == 0 and events[1]["seq"] == 1
        print("  TraceWriter/TraceReader: scalar round-trip preserves values and seq order")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_numpy_scalars_and_nan():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("x", {
            "a": np.float64(1.5), "b": np.int64(7), "c": np.bool_(True),
            "bad": float("nan"), "inf": float("inf"),
        })
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]
        assert payload["a"] == 1.5 and isinstance(payload["a"], float)
        assert payload["b"] == 7 and isinstance(payload["b"], int)
        assert payload["c"] is True
        assert payload["bad"] is None, "NaN must serialize as null, not crash json.dumps"
        assert payload["inf"] is None, "Inf must serialize as null, not crash json.dumps"
        print("  TraceWriter: numpy scalars unwrap to native types; NaN/Inf -> null, not a crash")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_array_becomes_histogram_not_raw_values():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        big = np.random.default_rng(0).normal(size=50000)
        w.emit("details", {"laplacian_mag": big})
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]
        hist = payload["laplacian_mag"]
        assert isinstance(hist, dict) and "counts" in hist and "edges" in hist
        assert hist["n"] == 50000
        assert sum(hist["counts"]) == 50000, "histogram must account for every sample"
        assert isinstance(hist["counts"][0], int)
        # The whole point: 50000 floats must NOT appear as a flat JSON list.
        raw = json.dumps(payload)
        assert raw.count(",") < 500, "a 50000-element array leaked through as raw values, not a histogram"
        print("  TraceWriter: large arrays become bounded histograms, never raw per-element values")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_degenerate_and_empty_arrays():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("x", {"constant": np.full(10, 3.0), "empty": np.zeros(0)})
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]
        assert payload["constant"]["n"] == 10 and payload["constant"]["min"] == 3.0
        assert payload["empty"]["n"] == 0 and payload["empty"]["counts"] == []
        print("  _histogram: constant-valued and empty arrays don't crash np.histogram's range check")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dataclass_recurses_generically():
    @dataclasses.dataclass
    class Inner:
        arr: object
        label: str

    @dataclasses.dataclass
    class Outer:
        inner: Inner
        total: float

    d = _tmpdir()
    try:
        w = TraceWriter(d)
        obj = Outer(inner=Inner(arr=np.array([1.0, 2.0, 3.0]), label="ref"), total=9.5)
        w.emit("x", {"report": obj})
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]["report"]
        assert payload["total"] == 9.5
        assert payload["inner"]["label"] == "ref"
        assert payload["inner"]["arr"]["n"] == 3, "nested dataclass's array field must ALSO become a histogram"
        print("  TraceWriter: nested dataclasses recurse field-by-field; arrays histogram at any depth")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_mesh_exported_not_inlined():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        box = trimesh.creation.box(extents=[1, 1, 1])
        w.emit("candidate_scored", {"which": "reference", "mesh": box})
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]
        mesh_info = payload["mesh"]
        assert mesh_info["vertex_count"] == len(box.vertices)
        assert mesh_info["face_count"] == len(box.faces)
        assert mesh_info["watertight"] is True
        glb_path = os.path.join(d, mesh_info["file"])
        assert os.path.exists(glb_path), "mesh must be exported to its own file, not inlined into the trace"
        assert os.path.getsize(glb_path) > 0
        raw = json.dumps(payload)
        assert "vertices" not in raw.lower() or len(raw) < 2000, "mesh data leaked into the JSON payload"
        print("  TraceWriter: trimesh.Trimesh exports to its own GLB; payload holds only a summary")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_mesh_export_includes_stl_sibling():
    """Every exported mesh must ALSO get an STL sibling alongside its GLB --
    GLB is what viewer.js renders in-browser, STL is the format someone
    actually wants when downloading a candidate to diff externally (see
    experiments/dpo_inference_steering/inspector/static/app.js's download links)."""
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        box = trimesh.creation.box(extents=[1, 1, 1])
        w.emit("candidate_scored", {"which": "reference", "mesh": box})
        w.close()
        mesh_info = TraceReader(d).read_all()[0]["payload"]["mesh"]
        assert mesh_info["file"].endswith(".glb")
        assert mesh_info["stl_file"].endswith(".stl")
        stl_path = os.path.join(d, mesh_info["stl_file"])
        assert os.path.exists(stl_path) and os.path.getsize(stl_path) > 0
        # Same base name, just a different extension -- so a caller can derive
        # one from the other without a second round-trip through the trace.
        assert mesh_info["file"].rsplit(".", 1)[0] == mesh_info["stl_file"].rsplit(".", 1)[0]
        print("  TraceWriter: every mesh export also gets a downloadable STL sibling alongside the GLB")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_none_mesh_handled():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("candidate_scored", {"which": "delta_initial", "mesh": None, "score": None})
        w.close()
        payload = TraceReader(d).read_all()[0]["payload"]
        assert payload["mesh"] is None and payload["score"] is None
        print("  TraceWriter: a failed decode (mesh=None) round-trips as null, not a crash")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_truncated_final_line_tolerated():
    """Simulates reading a trace WHILE its writer is mid-flush on the next
    line -- the exact race a live SSE tailer runs against a 20-minute DPO
    run. A truncated line must be treated as 'not there yet', not raised."""
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("a", {"x": 1})
        w.emit("b", {"x": 2})
        w.close()

        # Simulate a partial third line the writer hadn't finished flushing.
        with open(os.path.join(d, "trace.jsonl"), "a") as fh:
            fh.write('{"seq": 2, "t": 1.0, "type": "c", "payloa')  # deliberately cut mid-write

        events = TraceReader(d).read_all()
        assert len(events) == 2, "a truncated final line must be silently skipped, not raised or half-parsed"
        assert [e["type"] for e in events] == ["a", "b"]
        print("  TraceReader: a truncated final line (writer mid-flush) is tolerated, not a crash")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tail_from_resumes_at_exact_offset():
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("a", {"x": 1})
        w.close()

        reader = TraceReader(d)
        first_batch, offset = reader.tail_from(0)
        assert len(first_batch) == 1

        w2 = TraceWriter(d)  # reopen (as a live tailer would see a still-running writer)
        w2.emit("b", {"x": 2})
        w2.close()

        second_batch, offset2 = reader.tail_from(offset)
        assert len(second_batch) == 1 and second_batch[0]["type"] == "b", (
            "tail_from must resume exactly where it left off, seeing only NEW events"
        )
        assert offset2 > offset
        print("  TraceReader.tail_from: resumes from an exact byte offset, sees only new events")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_writer_resumes_seq_on_reopen():
    """A writer that reopens an existing trace.jsonl (process restart after
    a crash) must not restart seq at 0 and produce duplicate/ambiguous
    sequence numbers alongside the events already on disk."""
    d = _tmpdir()
    try:
        w1 = TraceWriter(d)
        w1.emit("a", {})
        w1.emit("b", {})
        w1.close()

        w2 = TraceWriter(d)
        evt = w2.emit("c", {})
        w2.close()
        assert evt["seq"] == 2, f"expected seq=2 after 2 prior events, got {evt['seq']}"
        print("  TraceWriter: reopening an existing trace resumes seq numbering, no collisions")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_is_finished_ignores_module_level_run_end():
    """A dpo_branch-level "run_end" event only means shape-SLat sampling
    finished -- print-prep still runs after it in the real pipeline and can
    still fail. is_finished() must key off the RUNNER's top-level
    session_end/error, not dpo_branch's own run_end, or the server would
    think a run was done while post-processing was still in flight."""
    d = _tmpdir()
    try:
        w = TraceWriter(d)
        w.emit("run_start", {})
        w.emit("run_end", {"returned_report": True})  # dpo_branch.py's module-level event
        assert not TraceReader(d).is_finished(), (
            "a module-level run_end must NOT be treated as the whole session finishing"
        )
        w.emit("printable_result", {"watertight": True})
        w.emit("session_end", {})  # runner.py's own top-level completion marker
        w.close()
        assert TraceReader(d).is_finished()
        print("  TraceReader.is_finished: keys off session_end, not dpo_branch's module-level run_end")
    finally:
        shutil.rmtree(d, ignore_errors=True)


TESTS = [
    test_roundtrip_scalars,
    test_numpy_scalars_and_nan,
    test_array_becomes_histogram_not_raw_values,
    test_degenerate_and_empty_arrays,
    test_dataclass_recurses_generically,
    test_mesh_exported_not_inlined,
    test_mesh_export_includes_stl_sibling,
    test_none_mesh_handled,
    test_truncated_final_line_tolerated,
    test_tail_from_resumes_at_exact_offset,
    test_writer_resumes_seq_on_reopen,
    test_is_finished_ignores_module_level_run_end,
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
