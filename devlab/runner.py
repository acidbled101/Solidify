"""
DPO Inspector runner: drives one DPO generation as a standalone process and
writes every event to <run_dir>/trace.jsonl via devlab.trace.TraceWriter.

Runs as a SUBPROCESS (see devlab/server.py), not a thread, because the DPO
branch step alone peaks at ~19GB unified memory (measured) -- a hard process
boundary means a crash or OOM-kill here can't take the dev server down with
it, and gives the server a real exit code / stderr tail to report instead of
a wedged thread.

Usage:
    python -m devlab.runner --run-dir traces/<id> --image path/to.png \\
        --pipeline-type 512 --steps 12 --seed 42

Writes trace.jsonl incrementally as generation proceeds (line-buffered, see
TraceWriter) and always ends the trace with either a "session_end" or
"error" event -- TraceReader.is_finished() keys off exactly those two, not
dpo_branch.py's own module-level "run_end" (which only covers the sampling
stage; print-prep still runs after it and can still fail -- see trace.py).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# This process's cwd is arbitrary (spawned by server.py) -- resolve paths
# relative to the repo root (this file's grandparent), not cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from devlab.trace import TraceWriter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run one DPO-branched generation, tracing every event.")
    p.add_argument("--run-dir", required=True, help="Directory to write trace.jsonl + exported meshes/output into.")
    p.add_argument("--image", required=True, help="Path to the input image.")
    p.add_argument("--pipeline-type", default="512", choices=["512", "1024"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-faces", type=int, default=1_000_000)
    p.add_argument("--t-branch", type=float, default=0.5)
    p.add_argument(
        "--num-branches", type=int, default=1,
        help="Fork/steer/resume this many times across the schedule instead of "
             "once (spread evenly across [0.3, 0.7], ignoring --t-branch when >1).",
    )
    p.add_argument("--branch-noise-scale", type=float, default=0.02)
    p.add_argument("--continuation-steps", type=int, default=2)
    p.add_argument("--num-delta-grad-steps", type=int, default=3)
    p.add_argument(
        "--best-of-n", type=int, default=0,
        help="Replace gradient steering with best-of-N rejection sampling: draw N "
             "perturbations, decode and judge each, keep whichever the judge prefers. "
             "0 (default) keeps the steering path. See DPOBranchConfig.best_of_n for "
             "the compute-parity arithmetic.",
    )
    p.add_argument("--dpo-beta", type=float, default=1.0)
    p.add_argument("--delta-max-norm-ratio", type=float, default=3.0)
    p.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    p.add_argument(
        "--vanilla-too", action="store_true",
        help="Also run a same-seed vanilla (non-DPO) generation for a same-run comparison "
             "(trellis_core.pipeline.run_generation). Roughly doubles wall-clock.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    trace = TraceWriter(run_dir)

    def on_event(event_type, payload):
        trace.emit(event_type, payload)

    try:
        trace.emit("session_start", {
            "image": os.path.basename(args.image),
            "pipeline_type": args.pipeline_type,
            "steps": args.steps,
            "seed": args.seed,
            "target_faces": args.target_faces,
            "model_id": args.model_id,
            "num_branches": args.num_branches,
            "best_of_n": args.best_of_n,
            "vanilla_too": args.vanilla_too,
            "pid": os.getpid(),
        })

        # MUST be the first trellis-related import (see trellis_core/__init__.py
        # / bootstrap.py) -- sets MPS-fallback env vars and sys.path before
        # torch is imported anywhere, including transitively.
        import trellis_core  # noqa: F401
        from PIL import Image as PILImage
        from trellis_core.pipeline import load_pipeline, WatchdogError
        from trellis_core import dpo_branch, dpo_generation

        trace.emit("stage", {"name": "load_pipeline", "model_id": args.model_id})
        t0 = time.time()
        pipeline = load_pipeline(args.model_id, device="mps")
        trace.emit("stage", {"name": "pipeline_loaded", "seconds": time.time() - t0})

        dpo_config = dpo_branch.DPOBranchConfig(
            t_branch=args.t_branch,
            num_branches=args.num_branches,
            branch_noise_scale=args.branch_noise_scale,
            continuation_steps=args.continuation_steps,
            num_delta_grad_steps=args.num_delta_grad_steps,
            best_of_n=args.best_of_n,
            dpo_beta=args.dpo_beta,
            delta_max_norm_ratio=args.delta_max_norm_ratio,
            decode_resolution=int(args.pipeline_type),
            seed=args.seed,
            verbose=True,
        )

        image = PILImage.open(args.image)
        output_prefix = str(run_dir / "output" / "model")

        result = dpo_generation.run_generation_with_dpo(
            pipeline, image,
            seed=args.seed, pipeline_type=args.pipeline_type, steps=args.steps,
            dpo_config=dpo_config, output_prefix=output_prefix,
            target_faces=args.target_faces, on_event=on_event,
        )

        summary = {
            "raw_vertex_count": result.raw_vertex_count,
            "raw_face_count": result.raw_face_count,
            "generation_seconds": result.generation_seconds,
            "postprocess_seconds": result.postprocess_seconds,
            "watertight": result.printable_result.watertight,
        }

        if args.vanilla_too:
            trace.emit("stage", {"name": "vanilla_comparison_start"})
            from trellis_core.pipeline import run_generation
            vt0 = time.time()
            vanilla_prefix = str(run_dir / "output" / "vanilla")
            vanilla = run_generation(
                pipeline, PILImage.open(args.image),
                seed=args.seed, pipeline_type=args.pipeline_type, steps=args.steps,
                target_faces=args.target_faces, no_texture=True,
                out_glb_path=vanilla_prefix + ".glb", out_obj_path=vanilla_prefix + ".obj",
            )
            trace.emit("vanilla_result", {
                "vertex_count": vanilla.vertex_count, "face_count": vanilla.face_count,
                "seconds": time.time() - vt0,
                "glb_path": vanilla_prefix + ".glb", "obj_path": vanilla_prefix + ".obj",
            })

        trace.emit("session_end", {"ok": True, "summary": summary})
        return 0

    except Exception as e:
        trace.emit("error", {
            "message": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "is_watchdog": type(e).__name__ == "WatchdogError",
        })
        print(f"[runner] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        trace.close()


if __name__ == "__main__":
    sys.exit(main())
