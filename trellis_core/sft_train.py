"""Supervised flow-matching fine-tune of the shape-SLat flow model.

WHAT THIS TRAINS AND WHY SFT RATHER THAN DPO
--------------------------------------------
With real printable meshes we have the *target*, not merely a comparison. DPO
exists because you have preferences and no ground truth; here the ground truth
is the encoded latent of a known-clean model. Supervised flow matching gives a
dense per-example signal instead of one bit per pair, which is what makes ~270
training examples a plausible dataset at all -- Diffusion-DPO used ~10^5 pairs.
It also needs no reference model and no preference machinery, so a step is one
forward and one backward rather than four forwards.

Objective, in TRELLIS.2's own path convention (read off flow_euler.py, not
assumed -- t runs 1 -> 0):

    x_t      = (1-t) x_0 + (sigma_min + (1-sigma_min) t) eps
    v_target = (1-sigma_min) eps - x_0
    L        = || v_theta(x_t, t, cond) - v_target ||^2

WHAT THIS CAN AND CANNOT BUY
----------------------------
Measured in trellis_core/vae_roundtrip.py: encoding a perfectly watertight mesh
and decoding it straight back preserves shape well (Chamfer 0.005) but does NOT
preserve watertightness (0/2, ~0.086% open edges, 953 components). The decoder
introduces its own cracks regardless of how clean the input was.

So training the flow model can move geometry -- wall thickness, overhang, bulk
proportion, flat bases -- because geometry survives the VAE. It cannot make
output watertight, because that defect is downstream of everything being
trained here. Watertightness belongs to post-processing (`printable.py`'s
repair), not to this objective. The eval below therefore tracks L_Th and L_OH
as the axes that can actually move, and reports watertight rate only as a
control that is expected to stay flat.

THE FAILURE MODE THE EVAL IS BUILT AROUND
-----------------------------------------
Fine-tuning toward a narrow domain improves printability metrics *while*
degrading prompt adherence, and a printability-only eval scores that as
success. This repo has already measured that exact shape: a 2-D experiment
tripled its reward while collapsing all 3000 samples onto one geometry. Eval
therefore always reports image fidelity and sample diversity alongside
printability, and any printability win with fidelity falling is a loss.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

from . import bootstrap  # noqa: F401  (env before torch)

import numpy as np
import torch

from .train_run import TrainRun, Control
from . import lora as lora_mod
from .dpo_branch import checkpointed_blocks
from .flow_dpo_objective import noise_latent, velocity_target, sample_timesteps, FlowDPOConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class SlatDataset:
    """Latents + conditioning, loaded lazily from the dataset build's output."""

    def __init__(self, root: str, ids: List[str]):
        self.root = root
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def load(self, fid: str, device: str):
        z = np.load(os.path.join(self.root, "latents", f"{fid}.npz"))
        coords = torch.from_numpy(z["coords"].astype(np.int32))
        feats = torch.from_numpy(z["feats"].astype(np.float32))
        cond = torch.from_numpy(np.load(os.path.join(self.root, "cond", f"{fid}.npy")).astype(np.float32))
        # SparseTensor coords carry a leading batch column.
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1)
        return coords.to(device), feats.to(device), cond.to(device)


def split_dataset(root: str, holdout: int, seed: int = 0):
    ids = [json.loads(l)["file_id"] for l in open(os.path.join(root, "manifest.jsonl"))]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return SlatDataset(root, ids[holdout:]), SlatDataset(root, ids[:holdout])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def flow_matching_loss_batch(model, samples, cfg: FlowDPOConfig, p_uncond: float = 0.0):
    """One forward/backward over a MINI-BATCH of sparse latents.

    SparseTensor already carries a batch index in coords[:, 0], so several
    objects can share one forward instead of being run one at a time and
    summed by gradient accumulation. That is the difference between N forward
    passes and one, and on MPS the per-launch overhead is a real share of a
    small model's step time.

    Objects have wildly different token counts (726 at p10, 4729 at p90), so
    batches are formed by a TOKEN budget rather than a fixed count -- a fixed
    count would either overflow memory on a batch of large meshes or waste it
    on a batch of small ones.

    The loss is the mean of per-object means, not the mean over all tokens.
    Token-mean would weight a 5000-token object seven times as heavily as a
    700-token one purely for being bigger.
    """
    from trellis2.modules import sparse as sp

    device = samples[0][1].device
    B = len(samples)
    t = sample_timesteps(B, cfg, device=device)

    feats_all, coords_all, tgt_all, idx_all, conds = [], [], [], [], []
    for b, (coords, feats, cond) in enumerate(samples):
        eps = torch.randn(feats.shape, device=device, dtype=feats.dtype)
        tb = t[b]
        feats_all.append(noise_latent(feats, eps, tb, cfg.sigma_min))
        tgt_all.append(velocity_target(feats, eps, cfg.sigma_min))
        c = coords.clone()
        c[:, 0] = b                      # batch index for this object
        coords_all.append(c)
        idx_all.append(torch.full((coords.shape[0],), b, dtype=torch.long, device=device))
        # conditioning dropout is per-object, as in the official CFG trainer
        conds.append(torch.zeros_like(cond) if (p_uncond > 0 and random.random() < p_uncond) else cond)

    x_t = torch.cat(feats_all, 0)
    v_tgt = torch.cat(tgt_all, 0)
    coords_cat = torch.cat(coords_all, 0)
    sample_index = torch.cat(idx_all, 0)
    cond_cat = torch.cat(conds, 0)

    st = sp.SparseTensor(feats=x_t.float(), coords=coords_cat)
    pred = model(st, (1000.0 * t).to(torch.float32), cond_cat)
    v_pred = pred.feats if hasattr(pred, "feats") else pred

    se = (v_pred.float() - v_tgt.float()).pow(2).mean(dim=1)     # per token
    per_obj = torch.zeros(B, device=device, dtype=se.dtype).index_add(0, sample_index, se)
    counts = torch.zeros(B, device=device, dtype=se.dtype).index_add(
        0, sample_index, torch.ones_like(se))
    loss = (per_obj / counts.clamp_min(1)).mean()
    return loss, [float(x) for x in t]


def flow_matching_loss(model, coords, feats, cond, cfg: FlowDPOConfig,
                       generator=None, p_uncond: float = 0.0):
    """One supervised flow-matching step on a single sparse latent."""
    from trellis2.modules import sparse as sp

    device = feats.device
    t = sample_timesteps(1, cfg, device=device, generator=generator)
    eps = torch.randn(feats.shape, device=device, dtype=feats.dtype, generator=generator)

    # Conditioning dropout, as the official CFG trainer does: with
    # probability p_uncond the conditioning is replaced by the null
    # embedding (zeros, which is what pipeline.get_cond uses for neg_cond).
    # Without it the unconditional branch drifts away from the base model
    # while inference still samples with CFG against it, so guidance quietly
    # degrades even as the conditional loss improves.
    if p_uncond > 0 and random.random() < p_uncond:
        cond = torch.zeros_like(cond)

    x_t = noise_latent(feats, eps, t.reshape(1, 1), cfg.sigma_min)
    v_tgt = velocity_target(feats, eps, cfg.sigma_min)

    # feats stay FLOAT32. The model is mixed precision: input_layer,
    # out_layer and t_embedder are fp32 while the 30 transformer blocks are
    # bf16, and `manual_cast` handles the transition internally. Handing it
    # bf16 features mismatches the fp32 input_layer and MPS aborts the
    # process with an MPSNDArrayMatrixMultiplication assertion rather than
    # raising. The pipeline's own sampler builds its noise with a plain
    # torch.randn, i.e. fp32, for exactly this reason.
    st = sp.SparseTensor(feats=x_t.float(), coords=coords)
    t_model = (1000.0 * t).to(device=device, dtype=torch.float32)
    pred = model(st, t_model, cond)
    v_pred = pred.feats if hasattr(pred, "feats") else pred
    return (v_pred.float() - v_tgt.float()).pow(2).mean(), float(t.item())


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, pipeline, ds: SlatDataset, cfg: FlowDPOConfig, *,
             device: str, n: int, steps: int, decode_resolution: int,
             sample_dir: Optional[str], judge_weights, n_render: int = 6) -> Dict:
    """Sample on held-out coords, decode, judge.

    Coords are taken from the held-out model itself rather than freshly sampled
    from the structure stage. That isolates what this LoRA actually changed:
    the structure model is untouched by training, so letting it re-sample would
    inject variance that has nothing to do with the thing being measured.
    """
    import os
    import trimesh
    from trellis2.modules import sparse as sp
    from . import geometric_judge

    out: Dict[str, List[float]] = {k: [] for k in
                                   ("heldout_loss", "L_OH", "L_Th", "L_Topo", "R_Detail", "S")}
    watertight, dispersion_feats, errors, sims = [], [], [], []
    nonmanifold, openrate, components = [], [], []
    sampler = pipeline.shape_slat_sampler
    norm = pipeline.shape_slat_normalization
    mean = torch.as_tensor(norm["mean"], device=device)
    std = torch.as_tensor(norm["std"], device=device)

    for i, fid in enumerate(ds.ids[:n]):
        coords, feats, cond = ds.load(fid, device)

        # held-out flow-matching loss, on a fixed draw so it is comparable
        # across evals rather than re-randomised each time
        g = torch.Generator(device="cpu").manual_seed(1234 + i)
        eps = torch.randn(feats.shape, generator=g).to(device)
        t = torch.full((1,), 0.5, device=device)
        x_t = noise_latent(feats, eps, t.reshape(1, 1), cfg.sigma_min)
        v_tgt = velocity_target(feats, eps, cfg.sigma_min)
        st = sp.SparseTensor(feats=x_t.float(), coords=coords)
        pred = model(st, (1000.0 * t).float(), cond)
        out["heldout_loss"].append(float((pred.feats.float() - v_tgt.float()).pow(2).mean()))

        # sample -> decode -> judge
        # Common random numbers: the sampling noise is seeded per held-out
        # object and reused at every eval, so successive evals are a PAIRED
        # comparison of the same trajectory rather than independent draws.
        # Without this, two evals of the SAME base model measured L_OH at
        # 0.2889 and 0.3871 -- a 34% swing from noise alone, which would bury
        # any training effect of plausible size. Same principle the judge
        # already uses for its thickness raycasts.
        g_noise = torch.Generator(device="cpu").manual_seed(9000 + i)
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], model.in_channels, generator=g_noise).to(device),
            coords=coords)
        # Use the pipeline's OWN sampler params. The shape SLat sampler is a
        # CFG + guidance-interval sampler whose _inference_model requires
        # guidance_strength and guidance_interval; passing only `steps` yields
        # a sample that is not what inference produces. These come from
        # pipeline.json (guidance_strength 7.5, rescale 0.5, interval
        # [0.6, 1.0], rescale_t 3.0) so eval measures the model as it will
        # actually be used.
        sp_params = {**pipeline.shape_slat_sampler_params}
        sp_params["steps"] = steps
        try:
            slat = sampler.sample(model, noise, cond=cond,
                                  neg_cond=torch.zeros_like(cond),
                                  verbose=False, **sp_params).samples
            slat = slat * std + mean
            meshes, _ = pipeline.decode_shape_slat(slat, decode_resolution)
            m = meshes[0]
            tm = trimesh.Trimesh(vertices=m.vertices.detach().float().cpu().numpy(),
                                 faces=m.faces.detach().cpu().numpy(), process=False)
        except Exception as e:
            # Never silent. A baseline that reports n_eval=0 with no reason is
            # how a 12-hour run produces no usable eval at all -- which is
            # exactly what happened on the first launch.
            errors.append(f"{fid}: {type(e).__name__}: {e}")
            continue
        if len(tm.faces) == 0:
            errors.append(f"{fid}: decoded mesh had 0 faces")
            continue

        rng = np.random.default_rng(7)   # fixed: eval-to-eval comparability
        s, _ = geometric_judge.score_mesh_detailed(tm, judge_weights, rng=rng)
        # Topology on the FULL mesh. The judge decimates before scoring and
        # that decimation is itself a large source of non-manifold edges
        # (measured ~50x inflation), so L_Topo cannot be used to rank models.
        from .topology_test import mesh_topology
        topo = mesh_topology(tm)
        nonmanifold.append(topo["nonmanifold_rate"])
        openrate.append(topo["open_rate"])
        components.append(topo["components"])
        out["L_OH"].append(s.overhang_penalty)
        out["L_Th"].append(s.thickness_penalty)
        out["L_Topo"].append(s.topology_penalty)
        out["R_Detail"].append(s.detail_reward)
        out["S"].append(s.total)
        watertight.append(1.0 if tm.is_watertight else 0.0)
        dispersion_feats.append(slat.feats.float().mean(0).cpu().numpy())

        # Image fidelity: silhouette IoU between the sampled mesh and the
        # conditioning image. Both are rendered by the same code with the same
        # camera (elevation 20, azimuth 35, bounding-sphere fit), so their alpha
        # masks are directly comparable. This is the guard against the failure
        # where printability improves because the model stopped following the
        # prompt -- a printability-only eval scores that as success.
        try:
            from . import render_mesh
            from PIL import Image
            ref_path = os.path.join(ds.root, "images", f"{fid}.png")
            if os.path.exists(ref_path):
                shot = render_mesh.render_views(tm, n_views=1, size=256,
                                                elevation_deg=20.0, azimuth_deg=35.0)[0]
                ref = Image.open(ref_path).convert("RGBA").resize((256, 256), Image.LANCZOS)
                a = np.asarray(shot)[..., 3] > 8
                b = np.asarray(ref)[..., 3] > 8
                union = (a | b).sum()
                if union:
                    sims.append(float((a & b).sum()) / float(union))
        except Exception as e:
            errors.append(f"{fid}: fidelity {type(e).__name__}: {e}")

        if sample_dir and i < n_render:
            # Export the mesh, not only a picture of it. Rendering and
            # discarding the geometry left a finished run with images and no
            # models, recoverable only from whichever adapters survived
            # pruning.
            try:
                tm.export(os.path.join(sample_dir, f"{fid}.glb"))
            except Exception as e:
                errors.append(f"{fid}: glb export {type(e).__name__}: {e}")
            try:
                from . import render_mesh
                # Fixed camera across every checkpoint. The dashboard shows the
                # same object at successive steps; if the viewpoint drifted, the
                # change you see would be the camera, not the model.
                render_mesh.render_views(tm, n_views=1, size=320,
                                         elevation_deg=20.0, azimuth_deg=35.0)[0].save(
                    os.path.join(sample_dir, f"{fid}.png"))
            except Exception:
                pass

    res = {k: (float(np.mean(v)) if v else None) for k, v in out.items()}
    res["watertight_rate"] = float(np.mean(watertight)) if watertight else None
    res["image_similarity"] = float(np.mean(sims)) if sims else None
    res["nonmanifold_rate"] = float(np.mean(nonmanifold)) if nonmanifold else None
    res["open_rate"] = float(np.mean(openrate)) if openrate else None
    res["components"] = float(np.mean(components)) if components else None
    res["n_eval"] = len(watertight)
    if errors:
        res["errors"] = errors[:6]
        print(f"[eval] {len(errors)} sample(s) failed: {errors[0]}", flush=True)
    if len(dispersion_feats) > 1:
        D = np.stack(dispersion_feats)
        # Mean pairwise distance between per-sample latent means: collapses
        # toward 0 if every prompt starts producing the same object.
        res["sample_dispersion"] = float(np.mean([
            np.linalg.norm(D[i] - D[j]) for i in range(len(D)) for j in range(i + 1, len(D))]))
    return res


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=time.strftime("sft-%Y%m%d-%H%M%S"))
    ap.add_argument("--data", default=os.path.join(REPO, "data", "thingi10k_sft"))
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation")
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--p-uncond", type=float, default=0.1,
                    help="conditioning dropout; matches the official trainer")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="skip latents larger than this (memory guard)")
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--eval-n", type=int, default=6)
    ap.add_argument("--eval-steps", type=int, default=12)
    ap.add_argument("--checkpoint-every", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--decode-resolution", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--eval-render", type=int, default=6,
                    help="how many held-out samples to render per eval")
    ap.add_argument("--no-baseline-eval", dest="baseline_eval", action="store_false",
                    help="skip the step-0 base-model eval that anchors every chart")
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false",
                    help="disable activation checkpointing (will swap-thrash on 32GB)")
    ap.add_argument("--empty-cache-every", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=200, help="linear warmup steps")
    ap.add_argument("--lr-min-ratio", type=float, default=0.02,
                    help="cosine floor as a fraction of peak lr")
    ap.add_argument("--token-budget", type=int, default=9000,
                    help="mini-batch is filled until total tokens exceed this")
    ap.add_argument("--max-batch", type=int, default=6)
    ap.add_argument("--grad-spike", type=float, default=5.0,
                    help="skip a batch whose grad norm exceeds this x running median")
    ap.add_argument("--keep-best", type=int, default=5,
                    help="how many best checkpoints to retain")
    ap.add_argument("--rank-metric", default="nonmanifold_rate",
                    help="eval key used to rank checkpoints (lower is better)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="run N steps and exit (no eval)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    run = TrainRun(args.run_id, args.runs_dir)
    run.write_meta({"kind": "sft_train", **vars(args)})
    run.status(step=0, total_steps=args.steps, state="loading")

    control = Control.read(run.dir)
    if control.version == 0:
        control = Control(version=1, eval_every=args.eval_every,
                          checkpoint_every=args.checkpoint_every, log_every=args.log_every)
        control.write(run.dir)

    train_ds, eval_ds = split_dataset(args.data, args.holdout, seed=args.seed)
    print(f"train {len(train_ds)}  holdout {len(eval_ds)}", flush=True)

    from . import pipeline as pipe_mod, geometric_judge
    print("loading pipeline ...", flush=True)
    t0 = time.time()
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    model = pipeline.models["shape_slat_flow_model_512"]
    # Place the flow model on the device once and keep it there: training
    # holds it resident, unlike inference which pages it in per sampling call.
    model.to(args.device)
    # low_vram is deliberately LEFT ALONE. It is not just a memory hint --
    # decode_shape_slat only moves the shape decoder onto the device when it is
    # set, so clearing it strands the decoder on CPU and every eval dies with
    # "Tensor for argument weight is on cpu but expected on mps". Nothing here
    # calls sample_shape_slat (the one path that would page the flow model back
    # to CPU), so the flow model stays put regardless.
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    adapted = lora_mod.apply_lora(model, rank=args.rank, alpha=args.alpha)
    trainable = lora_mod.count_trainable(model)
    print(f"LoRA on {len(adapted)} modules, {trainable/1e6:.2f}M trainable", flush=True)

    params = list(lora_mod.lora_parameters(model))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), eps=1e-8)
    # Schedule, sigma_min and weight decay follow TRELLIS.2's own training
    # config for this exact checkpoint (configs/gen/slat_flow_img2shape_dit_
    # 1_3B_512_bf16.json), not defaults invented here. Its trainer samples t
    # uniformly; the logit-normal schedule this module used before is the
    # SD3 convention, not the one this checkpoint was trained under.
    cfg = FlowDPOConfig(sigma_min=pipeline.shape_slat_sampler.sigma_min,
                        t_sampling="uniform", t_max=1.0)
    judge_weights = geometric_judge.JudgeWeights()

    def lr_at(step):
        """Linear warmup then cosine decay to `lr_min_ratio` of peak.

        The previous run used a constant 1e-4 for 8.7 epochs and destroyed
        itself at ~step 1100 with the gradient norm going from 0.05 to 332.
        Decay is the standard guard against that, and warmup keeps the first
        updates small while AdamW's moment estimates are still cold.
        """
        base = control.lr or args.lr
        if step <= args.warmup:
            return base * step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        p = min(max(p, 0.0), 1.0)
        return base * (args.lr_min_ratio +
                       (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    start_step = 0
    ckpt = run.latest_checkpoint()
    if ckpt:
        state = torch.load(os.path.join(ckpt, "adapter.pt"), map_location="cpu")
        lora_mod.load_lora_state_dict(model, state["lora"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"resumed from {ckpt} at step {start_step}", flush=True)

    total_steps = args.smoke or args.steps
    order = list(range(len(train_ds)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    cursor = 0
    ema_loss = None
    gn_hist, n_spikes = [], 0
    ckpt_scores = {}   # step -> rank metric, drives best-N retention
    t_start = time.time()
    skipped = 0
    samples_seen = start_step * args.accum
    n_train = len(train_ds)

    if start_step == 0 and not args.smoke and args.baseline_eval:
        print("baseline eval (adapters disabled) ...", flush=True)
        run.status(step=0, total_steps=total_steps, state="baseline eval")
        model.eval()
        with lora_mod.disabled_lora(model):
            base_res = evaluate(model, pipeline, eval_ds, cfg, device=args.device,
                                n=args.eval_n, steps=args.eval_steps,
                                decode_resolution=args.decode_resolution,
                                sample_dir=run.sample_dir(0), judge_weights=judge_weights,
                                n_render=args.eval_render)
        model.train()
        run.log("baseline", 0, **base_res)
        run.log("eval", 0, **base_res)
        print("[baseline] " + "  ".join(f"{k}={v:.4f}" for k, v in base_res.items()
                                        if isinstance(v, float)), flush=True)

    for step in range(start_step + 1, total_steps + 1):
        control = Control.read(run.dir, control)
        if control.stop:
            print("stop requested", flush=True)
            break
        while control.pause:
            run.status(step=step - 1, total_steps=total_steps, state="paused")
            time.sleep(5)
            control = Control.read(run.dir, control)
        if control.lr:
            for g in opt.param_groups:
                g["lr"] = control.lr

        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        bin_losses = {}
        step_tokens = 0

        # ---- assemble one mini-batch under a token budget --------------
        batch = []
        while len(batch) < args.max_batch:
            if cursor >= len(order):
                rng.shuffle(order); cursor = 0
            fid = train_ds.ids[order[cursor]]; cursor += 1
            coords, feats, cond = train_ds.load(fid, args.device)
            n = coords.shape[0]
            if n > args.max_tokens:
                skipped += 1
                del coords, feats, cond
                continue
            if batch and step_tokens + n > args.token_budget:
                cursor -= 1                      # put it back for the next step
                del coords, feats, cond
                break
            batch.append((coords, feats, cond))
            step_tokens += n

        lr_now = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr_now

        with checkpointed_blocks(model, args.grad_checkpointing):
            loss, t_vals = flow_matching_loss_batch(model, batch, cfg,
                                                    p_uncond=args.p_uncond)
            loss.backward()
        acc_loss = float(loss.detach())
        for tv in t_vals:
            bin_losses.setdefault(min(9, int(tv * 10)), []).append(acc_loss)
        for c, f, cd in batch:
            del c, f, cd
        del batch, loss

        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        # Drop, do not merely clip, a batch whose gradient is a wild outlier.
        # Clipping bounds one update but cannot stop AdamW's moments filling
        # with garbage across a sustained blow-up, which is how the previous
        # run destroyed itself. The median is over recent healthy steps, so
        # the threshold adapts as training settles.
        gn_hist.append(gn)
        if len(gn_hist) > 100:
            gn_hist.pop(0)
        med = statistics.median(gn_hist) if len(gn_hist) >= 20 else None
        spiked = med is not None and gn > args.grad_spike * max(med, 1e-8)
        if spiked:
            n_spikes += 1
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()

        # MPS holds freed blocks in a cache rather than returning them, and
        # latent sizes here vary 136..11685 tokens, so the cache grows to the
        # high-water mark of the largest sample seen and stays there. On a
        # 32GB machine that is exactly what tips a healthy run into swap, so
        # it is released periodically. Cheap next to a ~25s step.
        if args.device == "mps" and step % args.empty_cache_every == 0:
            torch.mps.empty_cache()

        ema_loss = acc_loss if ema_loss is None else 0.92 * ema_loss + 0.08 * acc_loss
        dt = time.time() - t_step
        mem = torch.mps.current_allocated_memory() / 1e9 if args.device == "mps" else 0.0
        peak = torch.mps.driver_allocated_memory() / 1e9 if args.device == "mps" else 0.0
        samples_seen += args.accum
        epoch = samples_seen / max(n_train, 1)

        if step % max(1, control.log_every or 1) == 0:
            run.log("train", step, loss=acc_loss, loss_ema=ema_loss,
                    lr=lr_now, grad_norm=gn, spiked=int(spiked), n_spikes=n_spikes,
                    batch=len(t_vals),
                    step_s=dt, mem_gb=mem, mem_peak_gb=peak, skipped=skipped,
                    epoch=epoch, samples_seen=samples_seen, tokens=step_tokens,
                    **{f"t_bin_{k}": float(np.mean(v)) for k, v in bin_losses.items()})
        elapsed = time.time() - t_start
        run.status(step=step, total_steps=total_steps, state="training",
                   elapsed_s=elapsed,
                   eta_s=(total_steps - step) * elapsed / max(step - start_step, 1))

        ee = control.eval_every or args.eval_every
        if not args.smoke and (control.eval_now or (ee and step % ee == 0)):
            if control.eval_now:
                control.eval_now = False
                control.write(run.dir)
            run.status(step=step, total_steps=total_steps, state="evaluating")
            sdir = run.sample_dir(step)
            model.eval()
            res = evaluate(model, pipeline, eval_ds, cfg, device=args.device,
                           n=args.eval_n, steps=args.eval_steps,
                           decode_resolution=args.decode_resolution,
                           sample_dir=sdir, judge_weights=judge_weights,
                           n_render=args.eval_render)
            model.train()
            run.log("eval", step, **res)
            if res.get(args.rank_metric) is not None:
                ckpt_scores[step] = float(res[args.rank_metric])
            print(f"[eval @ {step}] " + "  ".join(
                f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float)), flush=True)

        ce = control.checkpoint_every or args.checkpoint_every
        if not args.smoke and ce and step % ce == 0:
            d = run.ckpt_dir(step)
            torch.save({"lora": lora_mod.lora_state_dict(model),
                        "optimizer": opt.state_dict(), "step": step}, os.path.join(d, "adapter.pt"))
            run.mark_checkpoint_done(d)
            kept = run.prune_best_checkpoints(ckpt_scores, keep=args.keep_best)
            run.log("event", step, kept_checkpoints=kept,
                    rank_metric=args.rank_metric)

        if step % 10 == 0 or args.smoke:
            print(f"step {step}/{total_steps}  loss {acc_loss:.4f}  ema {ema_loss:.4f}  "
                  f"{dt:.1f}s  {mem:.1f}GB", flush=True)

    if not args.smoke:
        d = run.ckpt_dir(step)
        torch.save({"lora": lora_mod.lora_state_dict(model),
                    "optimizer": opt.state_dict(), "step": step}, os.path.join(d, "adapter.pt"))
        run.mark_checkpoint_done(d)
    run.status(step=step, total_steps=total_steps, state="finished",
               elapsed_s=time.time() - t_start, eta_s=0)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
