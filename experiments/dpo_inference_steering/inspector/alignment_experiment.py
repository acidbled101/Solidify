"""A controlled, n>1 preference-alignment experiment scored by the REAL judge.

Why this exists
---------------
The 3D pipeline gives us n=1. One image, one seed, one branch: S_ref 0.3124 ->
S_delta_final 0.4057. That is an anecdote, not evidence -- it cannot separate
"steering works" from "the perturbation got lucky", and at ~12 min/run we are
not going to get n=100 out of it this week.

So this builds a system where the SAME question can be asked at n=2000 in
under a minute, while keeping the parts that actually matter real:

  * the reward is trellis_core.geometric_judge -- the exact scoring code the
    pipeline uses, on real trimesh geometry, with the exact weights from
    dpo_branch._default_judge_weights()
  * the losses are trellis_core.flow_dpo -- the same functions the unit tests
    validate
  * the model is a real (small) flow-matching velocity field, trained by
    flow matching, aligned by DPO

What is toy is the *shape family*: a hollow frustum parameterised by two
numbers instead of a 4B-parameter SLat decoder. That is the deliberate trade
-- it buys statistical power in the one place the real pipeline cannot
provide any.

    z = (slope, wall)  ->  hollow frustum  ->  geometric_judge  ->  reward

`slope` is dr/dz of the outer wall, so slope > 1 leans more than 45 degrees
from vertical and trips the judge's real overhang term (its threshold is
n.g > cos(45deg), and for this surface n.g = sin(atan(slope))). `wall` is
wall thickness against the judge's real d_min. Both map to genuine FDM
failure modes rather than a made-up scalar.

Every mesh is normalised into the [-0.5, 0.5] cube before scoring, matching
what a decoded SLat mesh actually looks like (see
dpo_branch._default_judge_weights). Without that, a flared frustum is simply
a BIGGER object, its Laplacian energy is larger, and R_Detail rewards
overhang -- a size confound, not a judge property. Measured before the fix:
S rose monotonically 7.86 -> 57.38 as slope went 0.0 -> 2.0.

Run:  python experiments/dpo_inference_steering/inspector/alignment_experiment.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import torch
import torch.nn as nn
import trimesh

from experiments.dpo_inference_steering.inspector.trace import _histogram
from experiments.dpo_inference_steering import flow_dpo
from experiments.dpo_inference_steering import geometric_judge as gj
from experiments.dpo_inference_steering.dpo_branch import _default_judge_weights

# Latent box. z in [-1,1]^2 for the flow model; these map it to physical params.
SLOPE_RANGE = (-0.4, 2.4)
WALL_RANGE = (0.004, 0.075)
FRUSTUM_H, FRUSTUM_R0, SECTIONS = 0.5, 0.18, 40


# ---------------------------------------------------------------------------
# Latent -> geometry -> real judge reward
# ---------------------------------------------------------------------------


def z_to_params(z: np.ndarray) -> np.ndarray:
    """[-1,1]^2 -> (slope, wall). Clipped: the flow model can propose outside."""
    z = np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0)
    slope = SLOPE_RANGE[0] + (z[..., 0] + 1.0) * 0.5 * (SLOPE_RANGE[1] - SLOPE_RANGE[0])
    wall = WALL_RANGE[0] + (z[..., 1] + 1.0) * 0.5 * (WALL_RANGE[1] - WALL_RANGE[0])
    return np.stack([slope, wall], axis=-1)


def frustum_mesh(slope: float, wall: float) -> trimesh.Trimesh:
    """A closed hollow frustum standing on the build plate, unit-cube normalised.

    Closed (capped by an annulus at each end) so the judge's topology term
    measures something meaningful: an open tube would report a constant
    boundary-edge count regardless of (slope, wall) and the topology panel
    would be flat by construction rather than by finding.
    """
    h, r0, n = FRUSTUM_H, FRUSTUM_R0, SECTIONS
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    zs = np.array([0.0, h])
    verts, faces = [], []
    for ring_z in zs:
        r_out = max(r0 + slope * ring_z, 1e-3)
        r_in = max(r_out - wall, 5e-4)
        verts.append(np.stack([cos_a * r_out, sin_a * r_out, np.full(n, ring_z)], axis=1))
        verts.append(np.stack([cos_a * r_in, sin_a * r_in, np.full(n, ring_z)], axis=1))
    V = np.concatenate(verts, axis=0)  # [bot_out, bot_in, top_out, top_in]
    BO, BI, TO, TI = 0, n, 2 * n, 3 * n

    def quad(a0, b0, a1, b1, flip=False):
        """Bridge ring a (offset a0) to ring b (offset b0) with a quad strip."""
        for i in range(n):
            j = (i + 1) % n
            f1, f2 = [a0 + i, b0 + i, b0 + j], [a0 + i, b0 + j, a0 + j]
            faces.extend([f1[::-1], f2[::-1]] if flip else [f1, f2])

    quad(BO, TO, 0, 0)                 # outer wall
    quad(BI, TI, 0, 0, flip=True)      # inner wall (inward normals)
    quad(BO, BI, 0, 0)                 # bottom annulus
    quad(TO, TI, 0, 0, flip=True)      # top annulus

    mesh = trimesh.Trimesh(vertices=V, faces=np.array(faces), process=False)
    mesh.fix_normals()

    # Normalise into the [-0.5, 0.5] cube like a decoded SLat mesh, so scale
    # is not smuggled into R_Detail. See module docstring.
    extent = float(mesh.extents.max())
    if extent > 1e-9:
        mesh.vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) / extent
    return mesh


def overhang_split(mesh: trimesh.Trimesh, theta_crit_deg: float = 45.0,
                   bed_frac: float = 0.02) -> Tuple[float, float]:
    """Split L_OH into (build-plate contact, genuine wall overhang).

    geometric_judge.overhang_penalty() is purely normal-based: it penalises any
    downward-facing face, with no z_min or support test. So a cube resting flat
    on the build plate scores L_OH = 0.0732 -- entirely its bottom face, which
    every slicer scores zero because the plate supports it.

    That is not a nitpick here: measured on this shape family, 100% of L_OH
    comes from the bottom annulus at every slope the base distribution actually
    samples. Reporting the raw penalty as "overhang" would have credited
    alignment with a printability win that was really a change in footprint
    area. Faces whose centroid sits within `bed_frac` of z_min are attributed
    to the plate.

    NOTE: the missing bed test is a real defect in the shipping judge, not just
    in this experiment -- see report/flow_dpo_theory.md.
    """
    if len(mesh.faces) == 0:
        return 0.0, 0.0
    n_dot_g = -mesh.face_normals[:, 2]
    excess = np.maximum(0.0, n_dot_g - math.cos(math.radians(theta_crit_deg)))
    contrib = mesh.area_faces * excess
    z = mesh.triangles_center[:, 2]
    on_bed = z < (mesh.bounds[0][2] + bed_frac * float(mesh.extents[2] or 1.0))
    return float(contrib[on_bed].sum()), float(contrib[~on_bed].sum())


@dataclass
class RewardTerms:
    total: float
    detail: float
    overhang: float
    thickness: float
    topology: float
    overhang_bed: float       # part of L_OH that is build-plate contact (not overhang)
    overhang_true: float      # part that is genuine unsupported wall
    n_open: int
    n_nonmanifold: int
    watertight: bool


def score_params(slope: float, wall: float, weights: gj.JudgeWeights, seed: int = 0) -> RewardTerms:
    """Score one (slope, wall) with the real judge. ~7ms."""
    mesh = frustum_mesh(slope, wall)
    score, details = gj.score_mesh_detailed(mesh, weights, np.random.default_rng(seed))
    oh_bed, oh_true = overhang_split(mesh, weights.theta_crit_deg)
    return RewardTerms(
        total=float(score.total),
        detail=float(score.detail_reward),
        overhang=float(score.overhang_penalty),
        thickness=float(score.thickness_penalty),
        topology=float(score.topology_penalty),
        overhang_bed=oh_bed,
        overhang_true=oh_true,
        n_open=int(details.n_open),
        n_nonmanifold=int(details.n_nonmanifold),
        watertight=bool(mesh.is_watertight),
    )


class RewardField:
    """Real judge evaluations on a grid + bilinear interpolation.

    The interpolation is for speed only -- every grid node is a genuine
    score_mesh_detailed() call on real geometry. Alignment needs thousands of
    reward lookups; at ~7ms each, exact evaluation everywhere would dominate
    the runtime for no added fidelity, since the reward is a deterministic
    function of (slope, wall).

    grid_error() re-checks interpolated values against fresh exact evaluations
    at random off-node points, so the approximation is measured rather than
    assumed.
    """

    def __init__(self, weights: gj.JudgeWeights, res: int = 33, verbose: bool = True):
        self.weights, self.res = weights, res
        gz = np.linspace(-1.0, 1.0, res)
        self.gz = gz
        fields = {k: np.zeros((res, res)) for k in
                  ["total", "detail", "overhang", "thickness", "topology", "n_open", "n_nonmanifold"]}
        t0 = time.time()
        for i, z0 in enumerate(gz):
            for j, z1 in enumerate(gz):
                slope, wall = z_to_params(np.array([z0, z1]))
                r = score_params(float(slope), float(wall), weights)
                for k in fields:
                    fields[k][i, j] = getattr(r, k)
        self.fields = fields
        if verbose:
            print(f"  reward grid: {res}x{res} = {res*res} real judge calls in {time.time()-t0:.1f}s")

    def _bilinear(self, field: np.ndarray, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0)
        u = (z + 1.0) * 0.5 * (self.res - 1)
        i0 = np.clip(np.floor(u[..., 0]).astype(int), 0, self.res - 2)
        j0 = np.clip(np.floor(u[..., 1]).astype(int), 0, self.res - 2)
        fi, fj = u[..., 0] - i0, u[..., 1] - j0
        return (field[i0, j0] * (1 - fi) * (1 - fj) + field[i0 + 1, j0] * fi * (1 - fj)
                + field[i0, j0 + 1] * (1 - fi) * fj + field[i0 + 1, j0 + 1] * fi * fj)

    def __call__(self, z: np.ndarray, key: str = "total") -> np.ndarray:
        return self._bilinear(self.fields[key], z)

    def grid_error(self, n: int = 200, seed: int = 0) -> dict:
        """Measure the interpolation error against fresh exact judge calls."""
        rng = np.random.default_rng(seed)
        z = rng.uniform(-1, 1, size=(n, 2))
        approx = self(z, "total")
        exact = np.array([score_params(*z_to_params(zi), self.weights).total for zi in z])
        spread = float(exact.max() - exact.min())
        return {
            "rmse": float(np.sqrt(((approx - exact) ** 2).mean())),
            "reward_spread": spread,
            "rmse_pct_of_spread": float(np.sqrt(((approx - exact) ** 2).mean()) / max(spread, 1e-9) * 100),
            "rank_corr": float(np.corrcoef(approx, exact)[0, 1]),
        }


# ---------------------------------------------------------------------------
# The base (unaligned) generative distribution
# ---------------------------------------------------------------------------

# Six modes scattered across the latent box: some printable, some not. This
# stands in for "what an unaligned generator produces" -- a mix, which is
# exactly the situation preference alignment is supposed to improve.
BASE_MODES = np.array([
    [-0.65, 0.55], [-0.10, -0.60], [0.35, 0.30],
    [0.80, -0.35], [-0.45, -0.15], [0.55, 0.75],
])
MODE_STD = 0.13


def sample_base(n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(BASE_MODES), size=n)
    return np.clip(BASE_MODES[idx] + rng.normal(0, MODE_STD, size=(n, 2)), -1, 1)


def mode_entropy(z: np.ndarray) -> float:
    """Shannon entropy (nats) of the nearest-mode assignment.

    The reward-hacking / mode-collapse detector. A model that maximises reward
    by dumping all mass on one mode scores well on reward and badly here, and
    the loss curve shows neither. log(6) = 1.792 is full coverage.
    """
    d = ((z[:, None, :] - BASE_MODES[None, :, :]) ** 2).sum(axis=2)
    counts = np.bincount(d.argmin(axis=1), minlength=len(BASE_MODES)).astype(np.float64)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# The flow model
# ---------------------------------------------------------------------------


class TinyFlow(nn.Module):
    """v_theta(x, t): 2D velocity field. Small enough to train in seconds."""

    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([x, t], dim=1))


def train_flow_matching(model, data_sampler, steps=3000, batch=512, lr=2e-3, seed=0, verbose=True):
    """Standard conditional flow matching on the linear path."""
    torch.manual_seed(seed)
    path = flow_dpo.LinearPath()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        x1 = torch.tensor(data_sampler(batch, rng), dtype=torch.float32)
        x0 = torch.randn_like(x1)
        t = torch.rand(batch)
        xt = path.interpolate(x0, x1, t)
        target = path.target_velocity(x0, x1)
        loss = ((model(xt, t) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if verbose and (step + 1) % 1000 == 0:
            print(f"    fm step {step+1}/{steps}  loss={np.mean(losses[-200:]):.4f}")
    return losses


@torch.no_grad()
def sample_ode(model, n, steps=64, seed=0, return_traj=False, device="cpu"):
    """Deterministic ODE sampling, t: 0 -> 1. Optionally keep the whole path."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 2, generator=g, device=device)
    traj = [x.clone()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        x = x + model(x, t) * dt
        traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


# ---------------------------------------------------------------------------
# DPO alignment
# ---------------------------------------------------------------------------


def align_with_dpo(
    model, ref_model, reward_fn, *, steps=400, batch=256, lr=2e-4, beta=0.6,
    reference_free=False, seed=0, checkpoint_every=40, checkpoint_fn=None, verbose=True,
):
    """Align a trained flow model from pairwise preferences.

    Each iteration: draw pairs from the BASE model's own output distribution,
    label each pair by the real judge's reward, and take one DPO step. Drawing
    from the base model (not from the data) is what makes this preference
    *alignment* rather than supervised fine-tuning on filtered data -- the
    model is being corrected on its own samples, which is the setting the
    3D pipeline is in.
    """
    torch.manual_seed(seed)
    path = flow_dpo.LinearPath()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for step in range(steps):
        # Pairs from the frozen base model, labelled by the real judge.
        with torch.no_grad():
            cand = sample_ode(ref_model, batch * 2, steps=32, seed=seed * 10000 + step)
        z = cand.numpy()
        r = reward_fn(z)
        a, b = z[:batch], z[batch:]
        ra, rb = r[:batch], r[batch:]
        win = np.where((ra >= rb)[:, None], a, b)
        lose = np.where((ra >= rb)[:, None], b, a)

        x1_w = torch.tensor(win, dtype=torch.float32)
        x1_l = torch.tensor(lose, dtype=torch.float32)
        x0 = torch.randn(batch, 2)
        t = torch.rand(batch)
        xt_w, xt_l = path.interpolate(x0, x1_w, t), path.interpolate(x0, x1_l, t)
        tgt_w, tgt_l = path.target_velocity(x0, x1_w), path.target_velocity(x0, x1_l)

        v_w, v_l = model(xt_w, t), model(xt_l, t)
        if reference_free:
            loss = flow_dpo.dpo_loss_reference_free(v_w, tgt_w, v_l, tgt_l, beta=beta)
        else:
            with torch.no_grad():
                rv_w, rv_l = ref_model(xt_w, t), ref_model(xt_l, t)
            loss = flow_dpo.dpo_loss_velocity(v_w, rv_w, tgt_w, v_l, rv_l, tgt_l, beta=beta)

        opt.zero_grad(); loss.backward(); opt.step()

        if checkpoint_fn is not None and (step % checkpoint_every == 0 or step == steps - 1):
            history.append(checkpoint_fn(step, float(loss), model))
            if verbose:
                h = history[-1]
                print(f"    dpo step {step:4d}  loss={float(loss):.4f}  "
                      f"reward={h['reward_mean']:+.4f}  entropy={h['mode_entropy']:.3f}")
    return history


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm-steps", type=int, default=3000)
    ap.add_argument("--dpo-steps", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--grid-res", type=int, default=33)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "static", "concept_data.json"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    t_start = time.time()
    weights = _default_judge_weights()
    print(f"Judge weights (from dpo_branch._default_judge_weights): {weights}")

    # --- the reward field, from the real judge -----------------------------
    print("\n[1/6] Evaluating the real geometric judge over the latent box...")
    field = RewardField(weights, res=args.grid_res)
    err = field.grid_error(n=200, seed=1)
    print(f"  interpolation vs exact: RMSE {err['rmse']:.4f} "
          f"({err['rmse_pct_of_spread']:.2f}% of reward spread), corr {err['rank_corr']:.5f}")

    # How much does R_Detail dilute the printability signal? alpha is the
    # term that swamped everything before unit-cube normalisation, so sweep
    # IT rather than delta (the pipeline default already has alpha=beta=gamma=1;
    # only delta is softened, which is why an all-ones comparison is vacuous).
    print("  weight sensitivity -- corr(S, overhang penalty) vs the detail weight alpha:")
    probe_z = np.random.default_rng(3).uniform(-1, 1, size=(160, 2))
    probe_terms = [score_params(*z_to_params(z), weights) for z in probe_z]
    oh = np.array([e.overhang for e in probe_terms])
    th = np.array([e.thickness for e in probe_terms])
    det = np.array([e.detail for e in probe_terms])
    alpha_sweep = []
    for a in [0.0, 0.25, 1.0, 4.0]:
        s = a * det - (weights.beta * oh + weights.gamma * th)
        alpha_sweep.append({"alpha": a, "corr_overhang": float(np.corrcoef(s, oh)[0, 1])})
        print(f"    alpha={a:<5.2f} corr = {alpha_sweep[-1]['corr_overhang']:+.3f}"
              + ("   <-- pipeline default" if a == 1.0 else ""))
    weight_sensitivity = {
        "alpha_sweep": alpha_sweep,
        "note": ("alpha is the detail-reward weight. At alpha=0 the score is pure "
                 "printability. As alpha grows, R_Detail dilutes the overhang signal. "
                 "Before unit-cube normalisation this was catastrophic: S rose 7.86 -> "
                 "57.38 as slope went 0.0 -> 2.0, i.e. the judge preferred MORE overhang, "
                 "because a flared frustum is simply a larger object with more Laplacian "
                 "energy. That was a size confound in the shape parameterisation, not a "
                 "defect in the judge."),
    }

    def reward_fn(z):
        return field(z, "total")

    # --- base flow model ---------------------------------------------------
    print(f"\n[2/6] Training the base flow model ({args.fm_steps} steps)...")
    base = TinyFlow()
    fm_losses = train_flow_matching(base, sample_base, steps=args.fm_steps, seed=args.seed)

    ref = TinyFlow()
    ref.load_state_dict(base.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    # --- checkpoint metrics, measured with the real judge -------------------
    def checkpoint(step, loss, model):
        with torch.no_grad():
            z = sample_ode(model, 512, steps=32, seed=99).numpy()
        exact = [score_params(*z_to_params(zi), weights) for zi in z[:128]]
        return {
            "step": step,
            "loss": loss,
            "reward_mean": float(field(z, "total").mean()),
            "overhang": float(np.mean([e.overhang for e in exact])),
            "overhang_true": float(np.mean([e.overhang_true for e in exact])),
            "overhang_bed": float(np.mean([e.overhang_bed for e in exact])),
            "out_of_box_frac": float((np.abs(z) > 1).any(axis=1).mean()),
            "thickness": float(np.mean([e.thickness for e in exact])),
            "topology": float(np.mean([e.topology for e in exact])),
            "n_nonmanifold": float(np.mean([e.n_nonmanifold for e in exact])),
            "n_open": float(np.mean([e.n_open for e in exact])),
            "watertight_frac": float(np.mean([e.watertight for e in exact])),
            "mode_entropy": mode_entropy(z),
        }

    # --- align -------------------------------------------------------------
    print(f"\n[3/6] DPO alignment, reference-based ({args.dpo_steps} steps, beta={args.beta})...")
    aligned = TinyFlow()
    aligned.load_state_dict(base.state_dict())
    hist_ref = align_with_dpo(aligned, ref, reward_fn, steps=args.dpo_steps, beta=args.beta,
                              seed=args.seed, checkpoint_fn=checkpoint)

    print(f"\n[4/6] DPO alignment, REFERENCE-FREE ({args.dpo_steps} steps)...")
    aligned_free = TinyFlow()
    aligned_free.load_state_dict(base.state_dict())
    hist_free = align_with_dpo(aligned_free, ref, reward_fn, steps=args.dpo_steps, beta=args.beta,
                               reference_free=True, seed=args.seed, checkpoint_fn=checkpoint)

    # --- evaluate ----------------------------------------------------------
    print(f"\n[5/6] Evaluating (n={args.eval_n} per model, real judge on a subsample)...")

    def evaluate(model, tag, n_exact=400):
        z, traj = sample_ode(model, args.eval_n, steps=64, seed=1234, return_traj=True)
        zn = z.numpy()
        kappa = flow_dpo.path_curvature(traj)
        tort = flow_dpo.path_tortuosity(traj)
        kappa_n = flow_dpo.path_curvature_normalized(traj)
        exact = [score_params(*z_to_params(zi), weights) for zi in zn[:n_exact]]
        out = {
            "tag": tag,
            "reward": field(zn, "total"),
            "curvature": kappa.numpy(),
            "tortuosity": tort.numpy(),
            "curvature_normalized": kappa_n.numpy(),
            "out_of_box_frac": float((np.abs(zn) > 1).any(axis=1).mean()),
            "unique_clipped_slopes": int(len(np.unique(np.clip(zn[:, 0], -1, 1).round(6)))),
            "mode_entropy": mode_entropy(zn),
            "overhang": np.array([e.overhang for e in exact]),
            "overhang_true": np.array([e.overhang_true for e in exact]),
            "overhang_bed": np.array([e.overhang_bed for e in exact]),
            "thickness": np.array([e.thickness for e in exact]),
            "topology": np.array([e.topology for e in exact]),
            "n_nonmanifold": np.array([e.n_nonmanifold for e in exact], dtype=float),
            "watertight_frac": float(np.mean([e.watertight for e in exact])),
            "samples": zn,
            "traj": traj.numpy(),
        }
        print(f"  {tag:16s} reward {out['reward'].mean():+.4f}  kappa {kappa.mean():8.1f}  "
              f"tortuosity {tort.mean():.3f}  entropy {out['mode_entropy']:.3f}")
        print(f"  {'':16s} L_OH {out['overhang'].mean():.4f} "
              f"(bed {out['overhang_bed'].mean():.4f} + true {out['overhang_true'].mean():.4f})  "
              f"out-of-box {out['out_of_box_frac']:.1%}  distinct-slopes {out['unique_clipped_slopes']}")
        return out

    ev_base = evaluate(base, "base")
    ev_ref = evaluate(aligned, "dpo-reference")
    ev_free = evaluate(aligned_free, "dpo-ref-free")

    # Paired bootstrap on the reward difference: is the shift real?
    rng = np.random.default_rng(7)
    d = ev_ref["reward"] - ev_base["reward"]
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(2000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    cohens_d = float(d.mean() / (d.std() + 1e-12))
    print(f"  reward shift {d.mean():+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], Cohen's d {cohens_d:.2f}")

    # --- SDE/ODE gap on the trained field ----------------------------------
    print("\n[6/6] Measuring the ODE<->SDE marginal gap on the trained field...")
    gaps = []
    for sigma in [0.0, 0.15, 0.3, 0.6]:
        g = torch.Generator().manual_seed(5)
        x0 = torch.randn(4000, 2, generator=torch.Generator().manual_seed(11))
        gap = flow_dpo.sde_ode_marginal_gap(
            lambda x, t: base(x, t), x0, steps=64, sigma=sigma, generator=g
        )
        gaps.append(gap)
        print(f"  sigma={sigma:.2f}  per-sample RMSE {gap['per_sample_rmse']:.4f}  "
              f"mean shift {gap['mean_shift']:.4f}  std ratio {gap['std_ratio']:.3f}")

    # --- vector field for the dashboard ------------------------------------
    def field_arrows(model, t_val=0.5, res=15):
        gx = np.linspace(-1.6, 1.6, res)
        X, Y = np.meshgrid(gx, gx, indexing="ij")
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            v = model(pts, torch.full((pts.shape[0],), t_val)).numpy()
        return {"x": X.ravel().tolist(), "y": Y.ravel().tolist(),
                "u": v[:, 0].tolist(), "v": v[:, 1].tolist(), "t": t_val, "res": res}

    def thin_traj(traj, k=24):
        idx = np.linspace(0, traj.shape[1] - 1, k).astype(int)
        return traj[:, idx, :].transpose(1, 0, 2).tolist()

    data = {
        "meta": {
            "generated_by": "experiments/dpo_inference_steering/inspector/alignment_experiment.py",
            "judge_weights": {k: v for k, v in asdict(weights).items()},
            "fm_steps": args.fm_steps, "dpo_steps": args.dpo_steps,
            "eval_n": args.eval_n, "beta": args.beta, "seed": args.seed,
            "runtime_seconds": None,
            "grid_interpolation": err,
            "weight_sensitivity": weight_sensitivity,
            "provenance": (
                "Reward is trellis_core.geometric_judge on real trimesh geometry "
                "with dpo_branch._default_judge_weights(). Losses are "
                "trellis_core.flow_dpo. The SHAPE FAMILY is a 2-parameter frustum, "
                "not a TRELLIS.2 decode -- this measures the alignment mechanism at "
                "n>1, it does not measure the 4B pipeline."
            ),
        },
        "reward_grid": {
            "z": field.gz.tolist(),
            "total": field.fields["total"].tolist(),
            "overhang": field.fields["overhang"].tolist(),
            "thickness": field.fields["thickness"].tolist(),
            "modes": BASE_MODES.tolist(),
        },
        "vector_field": {
            "base": field_arrows(base),
            "aligned": field_arrows(aligned),
        },
        "trajectories": {
            "base": thin_traj(ev_base["traj"]),
            "aligned": thin_traj(ev_ref["traj"]),
        },
        "curvature": {
            "base": _histogram(ev_base["curvature"], bins=36),
            "aligned": _histogram(ev_ref["curvature"], bins=36),
            "base_mean": float(ev_base["curvature"].mean()),
            "aligned_mean": float(ev_ref["curvature"].mean()),
            "ref_free_mean": float(ev_free["curvature"].mean()),
            "tortuosity_base": float(ev_base["tortuosity"].mean()),
            "tortuosity_aligned": float(ev_ref["tortuosity"].mean()),
            "tortuosity_ref_free": float(ev_free["tortuosity"].mean()),
            "normalized_base": float(ev_base["curvature_normalized"].mean()),
            "normalized_aligned": float(ev_ref["curvature_normalized"].mean()),
            "caveat": (
                "Raw kappa scales as (displacement)^2, so mode collapse inflates it. "
                "Tortuosity (arc length / chord) and displacement-normalised kappa are "
                "scale-free and are the measures that answer 'did the paths straighten'. "
                "They disagree in DIRECTION with raw kappa on this run."
            ),
        },
        "reward_shift": {
            "base": _histogram(ev_base["reward"], bins=40),
            "aligned": _histogram(ev_ref["reward"], bins=40),
            "ref_free": _histogram(ev_free["reward"], bins=40),
            "base_mean": float(ev_base["reward"].mean()),
            "aligned_mean": float(ev_ref["reward"].mean()),
            "ref_free_mean": float(ev_free["reward"].mean()),
            "delta": float(d.mean()), "ci95": ci, "cohens_d": cohens_d,
            "n": int(args.eval_n),
        },
        "printability_history": {"reference_based": hist_ref, "reference_free": hist_free},
        "final_metrics": {
            tag: {
                "reward_mean": float(ev["reward"].mean()),
                "curvature_mean": float(ev["curvature"].mean()),
                "mode_entropy": ev["mode_entropy"],
                "overhang_mean": float(ev["overhang"].mean()),
                "overhang_true_mean": float(ev["overhang_true"].mean()),
                "overhang_bed_mean": float(ev["overhang_bed"].mean()),
                "tortuosity_mean": float(ev["tortuosity"].mean()),
                "out_of_box_frac": ev["out_of_box_frac"],
                "unique_clipped_slopes": ev["unique_clipped_slopes"],
                "thickness_mean": float(ev["thickness"].mean()),
                "topology_mean": float(ev["topology"].mean()),
                "n_nonmanifold_mean": float(ev["n_nonmanifold"].mean()),
                "watertight_frac": ev["watertight_frac"],
            }
            for tag, ev in [("base", ev_base), ("aligned", ev_ref), ("ref_free", ev_free)]
        },
        "max_mode_entropy": float(np.log(len(BASE_MODES))),
        "topology_limitation": (
            "NOT TESTABLE IN THIS FAMILY. The frustum is closed analytically, so every "
            "sample is watertight (n_open = n_nonmanifold = 0) at every point in the "
            "latent box, and the judge's topology term is identically zero. Emulating "
            "the real defect mechanism -- voxelise + marching cubes, as a decoded SLat "
            "mesh is produced -- does not help: a filled voxel grid is watertight by "
            "construction (measured: 0 non-manifold edges at wall thickness 0.006, "
            "0.020 and 0.050, at 100x the cost). So the brief's 'reduction of "
            "non-manifold edges over training steps' CANNOT be demonstrated here. The "
            "only real evidence available is the n=1 3D run, and it points the other "
            "way: that mesh came back watertight=False after repair."
        ),
        "sde_gap": gaps,
        "scatter": {
            "base": ev_base["samples"][:600].tolist(),
            "aligned": ev_ref["samples"][:600].tolist(),
        },
    }
    data["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"\nWrote {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB, "
          f"{data['meta']['runtime_seconds']}s total)")


if __name__ == "__main__":
    main()
