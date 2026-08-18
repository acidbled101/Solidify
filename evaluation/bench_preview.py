"""Measure what live previews actually cost.

The whole feature is conditional on "it doesn't add much computing power", so
this measures the added cost directly rather than estimating it.

Sampling is untouched by previews -- pred_x_0 is already computed at every step
-- so the added cost is exactly the sum of the preview decode times. That is
what this reports, alongside the total generation time it is a fraction of.

    python bench_preview.py [--resolution 128] [--at 0.5 0.85]
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")
from trellis_core import bootstrap  # noqa: F401
from trellis_core import progress as progress_mod
from trellis_core.pipeline import load_pipeline
from server import config

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/Users/scifablab/trellis-mac/dpo-worktree/3DBenchy.png")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--at", type=float, nargs="*", default=[0.5, 0.85])
    ap.add_argument("--pipeline-type", default=config.DEFAULT_PIPELINE_TYPE)
    ap.add_argument("--out", default="/tmp/preview_bench")
    args = ap.parse_args()

    from PIL import Image
    from trellis_core.pipeline import run_generation

    print(f"loading pipeline on {config.DEVICE} ...", flush=True)
    pipeline = load_pipeline(config.MODEL_ID, device=config.DEVICE)

    steps = []
    previews = []
    decode_times = []

    def on_step(stage, i, total):
        steps.append((stage, i, total, time.time()))

    def on_preview(path, step):
        previews.append((path, step))

    img = Image.open(args.image).convert("RGBA")

    t0 = time.time()
    with progress_mod.instrumented(
        pipeline,
        on_step=on_step,
        on_preview=on_preview,
        preview_at=args.at,
        preview_resolution=args.resolution,
        preview_dir=args.out,
    ):
        # Wrap _write_preview so its duration is attributed, not guessed.
        real_write = progress_mod._write_preview

        def timed(*a, **k):
            t = time.time()
            real_write(*a, **k)
            decode_times.append(time.time() - t)

        progress_mod._write_preview = timed
        try:
            run_generation(
                pipeline, img, seed=42, pipeline_type=args.pipeline_type,
                skip_texture=True, no_texture=True,
                out_glb_path=f"{args.out}/final.glb",
                out_obj_path=f"{args.out}/final.obj",
            )
        finally:
            progress_mod._write_preview = real_write
    total = time.time() - t0

    print("\n" + "=" * 62)
    stages = {}
    for stage, i, tot, _ in steps:
        stages.setdefault(stage, tot)
    print("sampler stages seen:", ", ".join(f"{s} ({n} steps)" for s, n in stages.items()))
    print(f"total sampling steps reported : {len(steps)}")
    print(f"generation wall clock         : {total:.0f}s")
    print(f"previews written              : {len(previews)} at resolution {args.resolution}")
    for p, s in previews:
        print(f"    step {s:>3}  {p}")
    if decode_times:
        added = sum(decode_times)
        print(f"preview decode time           : {added:.1f}s "
              f"({' + '.join(f'{d:.1f}' for d in decode_times)})")
        print(f"OVERHEAD                      : {100 * added / (total - added):.1f}% "
              f"of a run without previews")
    else:
        print("no previews were produced -- check preview_at / decode failures")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
