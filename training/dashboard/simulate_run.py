"""Write a synthetic run directory so the dashboard can be exercised without a GPU.

Exists because the dashboard must be trusted *before* an unattended weekend
run starts -- discovering a broken chart or a wrong field name at 3am, twelve
hours into training, wastes the run. This emits the exact schema
`sft_train.py` emits, at whatever speed you ask for.

    python -m devlab.simulate_run --run-id sim-001 --steps 400 --fast
    python -m devlab.simulate_run --run-id sim-live --live   # 1 step/s, for
                                                             # watching charts move
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train_run import TrainRun, Control  # noqa: E402


def base_loss_bin(i, step):
    """Loss is highest near t=1 (pure noise) and lowest near t=0."""
    return (0.15 + 1.4 * (i / 9) ** 1.6) * (0.4 + 0.6 * math.exp(-step / 150))


def _render_samples(run, step, rnd):
    """Render a few stand-in shapes so the evolution viewer has real frames."""
    try:
        import trimesh
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from trellis_core import render_mesh
    except Exception:
        return
    d = run.sample_dir(step)
    for i, name in enumerate(["obj_a", "obj_b", "obj_c"]):
        try:
            m = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
            # Deform progressively so successive checkpoints visibly differ.
            k = 0.30 * math.exp(-step / 200) + 0.02
            v = m.vertices.copy()
            rs = __import__("numpy").random.default_rng(1000 + i)
            v += rs.normal(0, k, v.shape)
            m = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
            render_mesh.render_views(m, n_views=1, size=320,
                                     elevation_deg=20.0, azimuth_deg=35.0)[0].save(
                os.path.join(d, f"{name}.png"))
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="sim-001")
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parent.parent / "runs"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--live", action="store_true", help="1 step/sec instead of instant")
    args = ap.parse_args(argv)

    run = TrainRun(args.run_id, args.runs_dir)
    run.write_meta({"kind": "sft_train", "simulated": True, "total_steps": args.steps, "lr": 1e-4})
    Control(version=1, eval_every=args.eval_every, checkpoint_every=100).write(run.dir)

    rnd = random.Random(0)
    prev_ema = None
    t0 = time.time()
    n_train = 270

    def eval_row(step, p):
        return dict(
            heldout_loss=(0.35 + 0.9 * math.exp(-step / 120)) * 1.05 + rnd.gauss(0, 0.02),
            L_OH=0.095 - 0.030 * p + rnd.gauss(0, 0.003),
            L_Th=0.046 - 0.021 * p + rnd.gauss(0, 0.002),
            L_Topo=0.150 - 0.004 * p + rnd.gauss(0, 0.004),
            R_Detail=0.21 - 0.06 * p + rnd.gauss(0, 0.006),
            watertight_rate=max(0.0, 0.02 + 0.01 * p + rnd.gauss(0, 0.005)),
            image_similarity=0.82 - 0.10 * p + rnd.gauss(0, 0.01),
            chamfer_vs_base=0.004 + 0.010 * p + rnd.gauss(0, 0.0006),
            mode_entropy=1.79 - 0.45 * p + rnd.gauss(0, 0.02),
            sample_dispersion=0.31 - 0.08 * p + rnd.gauss(0, 0.01),
            n_eval=6)

    # Baseline: the base model before any training. Anchors every chart.
    base = eval_row(0, 0.0)
    run.log("baseline", 0, **base)
    run.log("eval", 0, **base)
    _render_samples(run, 0, rnd)
    for step in range(1, args.steps + 1):
        # Loss: exponential decay toward a floor, plus heteroscedastic noise
        # that shrinks with the loss -- what a real flow-matching curve looks
        # like, so the EMA and axis autoscaling get a realistic workout.
        base = 0.35 + 0.9 * math.exp(-step / 120)
        loss = base * (1 + rnd.gauss(0, 0.09))
        ema = loss if step == 1 else 0.92 * prev_ema + 0.08 * loss
        prev_ema = ema
        samples = step * 4
        run.log("train", step,
                loss=loss, loss_ema=ema,
                lr=1e-4 * (0.5 * (1 + math.cos(math.pi * step / args.steps))),
                grad_norm=abs(rnd.gauss(1.2, 0.4)),
                step_s=rnd.uniform(60, 90),
                mem_gb=rnd.uniform(19, 24), mem_peak_gb=rnd.uniform(24, 28),
                epoch=samples / n_train, samples_seen=samples,
                **{f"t_bin_{i}": base_loss_bin(i, step) for i in range(10)})

        if step % args.eval_every == 0:
            p = step / args.steps
            _render_samples(run, step, rnd)
            run.log("eval", step,
                    heldout_loss=base * 1.05 + rnd.gauss(0, 0.02),
                    L_OH=0.095 - 0.030 * p + rnd.gauss(0, 0.003),
                    L_Th=0.046 - 0.021 * p + rnd.gauss(0, 0.002),
                    L_Topo=0.150 - 0.010 * p + rnd.gauss(0, 0.004),
                    R_Detail=0.21 - 0.06 * p + rnd.gauss(0, 0.006),
                    watertight_rate=min(0.35, 0.02 + 0.30 * p) + rnd.gauss(0, 0.01),
                    image_similarity=0.82 - 0.10 * p + rnd.gauss(0, 0.01),
                    chamfer_vs_base=0.004 + 0.010 * p + rnd.gauss(0, 0.0006),
                    mode_entropy=1.79 - 0.45 * p + rnd.gauss(0, 0.02),
                    sample_dispersion=0.31 - 0.08 * p + rnd.gauss(0, 0.01),
                    n_eval=8)

        elapsed = time.time() - t0
        run.status(step=step, total_steps=args.steps, state="training",
                   elapsed_s=elapsed,
                   eta_s=(args.steps - step) * (elapsed / max(step, 1)))
        if args.live:
            time.sleep(1.0)

    run.status(step=args.steps, total_steps=args.steps, state="finished",
               elapsed_s=time.time() - t0, eta_s=0)
    print(f"wrote {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
