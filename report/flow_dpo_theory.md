# Preference alignment for continuous flow models

Companion note to `report.tex` and `trellis_core/flow_dpo.py`. Covers two
questions: *why* DPO can be applied to a flow model at all, and *whether*
preference alignment is the right tool for FDM printability. Every quantitative
claim below is produced by `devlab/alignment_experiment.py` (seed 0, 104 s) or
`trellis_core/flow_dpo_test.py` (14 tests), and is reproducible from a clean
checkout.

Findings that contradict the motivating hypothesis are kept, not dropped. Two
of the four claims this note set out to support did not survive measurement.

---

## 1. Why DPO applies to a flow model

### 1.1 The obstruction

Flow matching learns a velocity field $v_\theta(x,t)$ whose ODE

$$\frac{dx}{dt} = v_\theta(x,t), \qquad x_0 \sim \mathcal N(0,I)$$

transports noise to data. DPO's derivation requires a *policy* — a conditional
distribution with a finite, differentiable likelihood ratio
$\pi_\theta/\pi_{\text{ref}}$. The ODE's induced measure over trajectories is a
Dirac delta given $x_0$: $\log \pi_\theta(\tau)$ is $\pm\infty$ and the DPO
log-ratio is undefined pointwise. **DPO cannot be applied to an ODE as stated.**
This is a real obstruction, not a technicality to wave through.

### 1.2 The mapping

The ODE is one member of a family of processes sharing its time marginals
$p_t$. For any $\sigma_t \ge 0$,

$$dx = \Big[v(x,t) + \tfrac{\sigma_t^2}{2}\, s(x,t)\Big]dt + \sigma_t\, dW,
\qquad s(x,t) = \nabla_x \log p_t(x) \tag{1}$$

has the same $p_t$ as the ODE. One line of Fokker–Planck: for $dx = f\,dt +
\sigma dW$, $\partial_t p = -\nabla\!\cdot\!(fp) + \tfrac{\sigma^2}{2}\Delta p$,
and substituting $f = v + \tfrac{\sigma^2}{2}\nabla\log p$ gives

$$-\nabla\!\cdot\!(vp) - \tfrac{\sigma^2}{2}\Delta p + \tfrac{\sigma^2}{2}\Delta p
= -\nabla\!\cdot\!(vp),$$

the continuity equation the ODE satisfies. $\sigma_t = 0$ recovers the ODE
exactly, so the ODE is the $\sigma\to0$ member rather than a separate case
(`test_sigma_zero_is_the_ode`).

Discretising (1) gives a Gaussian transition kernel

$$\pi_\theta(x_{t+\Delta}\mid x_t) = \mathcal N\big(x_t + \text{drift}_\theta \Delta,\ \sigma_t^2\Delta I\big),$$

so $\log\pi_\theta$ is a finite quadratic and the DPO log-ratio becomes a
difference of squared errors. **That is the entire trick.**

### 1.3 Why it costs nothing on the linear path

Equation (1) needs the score, which normally means a second network. For the
linear/rectified path $x_t=(1-t)x_0+tx_1$ — the path TRELLIS.2's SLat sampler
uses — it does not. Conditioning on $x_1$, $x_t \sim \mathcal N(tx_1,(1-t)^2I)$,
and since $x_t = x_0 + tv$:

$$s(x,t) = -\frac{x_t - t\,v(x,t)}{1-t} \tag{2}$$

The score is an *exact affine function of the velocity the model already
predicts*. Taking $\mathbb E[\,\cdot\mid x_t]$ of both sides lifts it from the
conditional to the marginal, which is what a flow-matching-trained $v_\theta$
regresses to, so the identity survives the lift.

Verified against a closed-form Gaussian marginal, not spot-checked:
`test_score_identity_is_exact` holds to $<10^{-5}$ across $t \in [0,0.9]$.

### 1.4 Measured: same law, different paths

The equivalence is at the level of **marginals, not path measures**. Integrating
the same trained field both ways (`sde_ode_marginal_gap`, n=4000):

| σ | per-sample RMSE | mean shift | std ratio |
|---|---|---|---|
| 0.00 (ODE) | 0.0000 | 0.0000 | 1.000 |
| 0.15 | 0.1583 | 0.0068 | 0.997 |
| 0.30 | 0.2873 | 0.0082 | 0.985 |
| 0.60 | 0.4839 | 0.0145 | 0.964 |

Per-sample RMSE is large (different trajectories) while mean shift ≈ 0 and std
ratio ≈ 1 (same distribution). **Both halves matter.** Asserting only the second
would pass for a broken SDE that silently collapsed to the ODE, which is why
`test_sde_preserves_marginals` asserts both directions.

*Consequence.* Marginal-level equivalence is exactly enough for an **endpoint**
reward — a mesh judge scores final geometry, not the route — and **not enough**
for any path-dependent reward. Our judge is an endpoint reward, so the gap
closes for us. It would not close for, say, a reward penalising intermediate
self-intersection.

### 1.5 A cost found by testing, not derivation

Substituting (2), the drift is affine in $v$ with slope

$$k(t) = 1 + \frac{\sigma^2 t}{2(1-t)}$$

The practical velocity-space loss is obtained from the true trajectory loss by
absorbing a prefactor containing $k(t)^2$ into $\beta$. Since $k(t)\to\infty$ as
$t\to1$:

> **A single global $\beta$ is exactly correct at only one timestep whenever
> $\sigma>0$.** At $\sigma=0$, $k\equiv1$ and the reduction is exact everywhere.

Measured at $\sigma=0.7$: $k(0.1)=1.03$, $k(0.5)=1.25$, $k(0.9)=3.20$,
$k(0.99)=25.26$ (`test_beta_conversion_is_timestep_dependent`). The noise that
makes DPO applicable is the same noise that makes its cheap form
$t$-inhomogeneous. This was found because the first draft of
`test_velocity_form_matches_trajectory_form` failed by a factor of ~7000 — it
would not have appeared in a derivation done on paper and accepted.

### 1.6 On "no reward model"

DPO's implicit reward $r(x) = \beta\log(\pi_\theta/\pi_{\text{ref}})$ makes the
optimal KL-regularised policy $\pi^*(x)\propto\pi_{\text{ref}}(x)e^{r(x)/\beta}$
reachable by optimising $\pi$ directly. There is no separate network to overfit
and no reward-model/policy distribution mismatch, because the reward is a
*reparameterisation of the policy* rather than an estimate of something external.

The sharp edge, stated because it is usually omitted: the implicit reward is
defined **relative to a reference**, so it inherits the reference's pathologies
rather than escaping them. For opposed targets, $r_w > r_l$ reduces exactly to
$\langle v_{\text{ref}}, v_w\rangle < \|v_w\|^2$ — a condition on the
*reference*, not on how opposed the preferences are. With a neutral reference
the ranking is a per-sample guarantee (100%); with an arbitrary one it holds
only in distribution (91.4% measured), and the derived condition predicts every
failure exactly (`test_implicit_reward_ranks_winner_above_loser`).

### 1.7 Reference-free

Dropping $\pi_{\text{ref}}$ (SimPO-style, with a margin $\gamma$) removes a
frozen model copy and a backward-enabled forward pass per pair. On this repo's
hardware that is not academic: one such forward through the SLat model retains
~9.5 GB of activations, and four per gradient step produced the 36.27 GiB MPS
OOM that `checkpointed_blocks()` was written to fix.

Analytically, both objectives differentiate the same bracket and only
$\text{err}(\theta,\cdot)$ depends on $\theta$, so

$$\frac{\partial\mathcal L}{\partial\theta} = c_i\cdot 2(v^*_l - v^*_w),\qquad c_i>0$$

for both. **Per-sample gradient directions are exactly parallel** (measured
cos = 1.000000); the reference only changes the positive per-sample weight
$c_i$, i.e. which pairs get attention. Reference-free is a *reweighting*, not a
different objective.

What it costs: $\pi_{\text{ref}}$ is what bounds the KL to the base model.
Without it, $\gamma$ and early stopping do work a principled KL used to do.
That is a weaker guarantee, not an equivalent one.

---

## 2. Printability: where preference alignment is and isn't the right tool

### 2.1 The one strong argument

**Preference pairs encode constraints that have no differentiable form.**
Non-manifold edge count is an integer-valued, discontinuous functional of the
mesh; its gradient is zero almost everywhere. You cannot backpropagate through
it. You *can* rank two meshes by it. Same for "did this print succeed" — the
ground-truth signal is a binary outcome of a physical process with no
computational graph at all.

This is the honest case for DPO here, and it is genuinely strong. It is also
narrower than the brief implies.

### 2.2 The counter-argument, which mostly wins

Where a differentiable relaxation exists, a direct physics loss is **strictly
better**: denser signal, no preference-pair sample complexity, no reward model
to hack. And relaxations exist for most of what we actually penalise:

- overhang → smooth $\max(0, n\cdot g - \cos\theta_{\text{crit}})$, already
  differentiable in the face normals
- thickness → SDF-based, differentiable
- topology → **not** differentiable; this is the genuine DPO case

So of the judge's four terms, three admit direct differentiable losses. The
composite $S = \alpha R_{\text{Detail}} - (\beta L_{OH} + \gamma L_{Th} +
\delta L_{Topo})$ uses preference machinery for terms that did not need it.

### 2.3 Measured: what alignment actually did

n=3000 per model, reward from the real `geometric_judge` with the pipeline's own
`_default_judge_weights()`:

> **Correction.** An earlier draft of this section reported two results that an
> adversarial review overturned. Both were measurement errors on my side, and
> both are described here rather than quietly replaced. The numbers below are
> the corrected run.

| model | reward | L_OH (raw) | L_OH true wall | mode entropy | tortuosity | out-of-box |
|---|---|---|---|---|---|---|
| base | 7.367 | 0.0838 | 0.0686 | 1.790 | 2.420 | 1.6% |
| reference-based DPO | 19.460 | 0.0606 | **0.0000** | 0.317 | 1.457 | **100%** |
| reference-free DPO | 19.468 | 0.0579 | **0.0000** | 0.417 | 1.436 | **100%** |

Reward shift Δ=12.094, 95% CI [11.91, 12.29], Cohen's d = 2.28, n=3000.
**The alignment mechanism works.** Four caveats, each measured:

**(a) The reported overhang improvement was mostly not overhang.**
`geometric_judge.overhang_penalty` (`geometric_judge.py:119`) is purely
normal-based — no z_min, no support test — so a cube resting flat on the build
plate scores $L_{OH} = 0.0732$, entirely its bottom face, which every slicer
scores zero. Splitting the penalty: **100% of the aligned model's $L_{OH}$ is
build-plate contact**. The genuine result is better than the one first
reported — true wall overhang goes $0.0686 \to 0.0000$ by step 120 — but the
raw figure ($0.0838 \to 0.0606$, "27%") was measuring footprint area. *This is
a real defect in the shipping judge, not only in this experiment.*

**(b) Reward hacking is present, in thickness rather than overhang.** $L_{Th}$
bottoms at 0.0279 at step 120 and then degrades **7.4×** to 0.2069 by step 499,
while reward buys its last 9% (17.77 → 19.46) and the loss curve looks healthy
throughout. Measured: $\mathrm{corr}(S, L_{OH})$ is $-0.809$ at $\alpha=0$ but
only $-0.428$ at the pipeline's $\alpha=1$ — $R_{\text{Detail}}$ dilutes the
printability signal by roughly half.
*Practical consequence: early-stop on a held-out physical metric, never on the
reward. Here that means stopping at step 120, which keeps every real gain.*

**(c) The optimum is a parameterisation artifact.** Entropy falls 1.790 → 0.317
of a possible 1.792, and worse: **100% of aligned samples land outside the
latent box** (base: 1.6%), where `z_to_params` clips them. All 3000 collapse to
**one** distinct geometry, sitting on the `max(..., 1e-3)` clamp in
`frustum_mesh`. DPO found the argmax of the proxy; the argmax is an artifact of
how the proxy was parameterised. The mechanism is confirmed; the *win* is not.
Note this happened under the **reference-based** objective — the KL anchor did
not prevent it, because the velocity-space surrogate contains no term
penalising distance from the reference.

**(d) DPO did straighten the paths — the first draft's claim to the contrary
was a metric error.** `path_curvature` measures $\|x''\|^2$, which scales as
(displacement)², and collapsing modes onto a distant point lengthens every path.
Raw $\kappa$ went 8.8 → 325.5 (37× "worse"), which the first draft reported as
refuting the straightening claim. Both scale-free measures disagree in
*direction*: tortuosity (arc length ÷ chord) went **2.420 → 1.457, i.e. 1.66×
straighter**, and displacement-normalised curvature agrees. The honest reading
is that the paths did straighten, but substantially *because* they now all end
in the same place — this is not evidence that DPO substitutes for reflow.
`test_tortuosity_is_scale_free_and_curvature_is_not` now pins the invariance.

### 2.4 Claims I could not test

- **"Reduction of non-manifold edges over training."** The frustum family is
  closed analytically, so every sample is watertight and the topology term is
  identically zero. Emulating the real defect mechanism (voxelise → marching
  cubes) does not help: a filled voxel grid is watertight by construction
  (measured: 0 non-manifold edges at wall thickness 0.006/0.020/0.050, at 100×
  the cost). **Untestable here.** The only real evidence is the n=1 3D run, and
  it points the other way: `watertight=False` after repair.

- **"Texture-geometry consistency."** TRELLIS.2's SLat *is* a joint
  geometry+appearance representation over shared sparse voxels, so a single
  latent edit moves both coherently — that architectural claim is true. But our
  pipeline steers only the **shape** SLat. The claim overreaches for what is
  implemented.

- **"Without requiring extensive post-processing."** Directly contradicted by
  our own measurement: the successful 3D run produced `watertight=False` and
  still required repair and decimation from 1M faces.

- **Transfer to the 3D pipeline.** Not shown. The 2-D result establishes the
  mechanism, not that it carries to 4B parameters and a 2858-voxel SLat.

### 2.5 Where multi-stage lift-off does lose

One part of the brief's premise survives intact: image → multiview → mesh
pipelines accumulate error across stages with no single differentiable path from
the final geometry back to the generative parameters. A single-stage latent flow
has exactly that path, which is what makes any gradient-based steering — DPO or
a direct physics loss — possible at all. That is an argument for **single-stage
latent generation**, not specifically for DPO.

---

## 3. The pipeline's own answer, pooled across every run

Section 2 measured the alignment *mechanism* on a toy. This measures the thing
that actually ships, on the traces already on disk, with **zero new GPU time**.

`devlab/metrics.py` flattens every run into per-fork rows. Each fork is an
independent trial of the same question, which is what turns a pile of n=1 runs
into a usable sample. The decisive column is `verdict_flipped`: `dpo_branch`
already records the judge's verdict on the **random** perturbation
(`delta_branch_won_initial`) and again after the **gradient** steps
(`delta_branch_won_final`). If those never disagree, the gradient is adding
nothing the random draw did not already provide.

Across **10 scored forks from 6 completed real runs**
(synthetic demo traces excluded — including them alone moves the mean from
−0.005 to −0.030):

| quantity | value |
|---|---|
| mean gain from the random perturbation | **+0.0067** |
| mean gain from gradient steering | **-0.0049 ± 0.0115** (SE) |
| steering helped / hurt | 4 / 6 forks |
| verdict flipped by steering | 4/10 (rate 0.4) |
| statistically distinguishable from zero? | **no** |

**The gradient's contribution is not distinguishable from noise.** Its point
estimate is slightly negative, it helped fewer forks than it hurt, and the
standard error is more than twice the effect. The random perturbation, by
contrast, is consistently positive.

Three things this does *not* say. It does not say the gradient is useless —
n=10 cannot rule out a real effect this small. It does not say the output got
worse: the re-rank gate only resumes from the steered candidate when the judge
prefers it, so a negative steering gain costs compute, not quality. And it does
not indict the DPO formulation — it indicts this particular proxy
(`slat_detail_proxy`, whose correlation with the judge's actual reasoning is
explicitly flagged as unknown in `dpo_branch.py`'s own docstring).

What it does say is that the ~3 extra backward passes per fork are currently
unjustified, and that **best-of-N sampling is the baseline to beat** before any
more is invested here. At `continuation_steps=2` and 3 gradient steps a fork
costs 10 flow forwards, 3 of them with backward; best-of-5 costs 10 forwards
and zero backwards, needs no gradient checkpointing, and can use the fast conv
backend throughout.

Reproduce: `python devlab/metrics.py --print`, or open the Concept Proof page's
"Every run, pooled" panel. Raw rows are in `devlab/traces/metrics_branches.csv`.

---

## 4. What would actually settle it

1. **Steering vs. best-of-N at matched compute.** Now the top priority, given
   §3: the gradient's measured contribution is −0.005 ± 0.011 while costing 3
   backward passes per fork. Implement best-of-N by calling `_continue` N times
   with `enable_grad=False` and ranking with the existing
   `rank_candidates_detailed`; at N=5 that is the same 10 forwards with no
   backward pass, no gradient checkpointing, and the fast conv backend
   throughout. If it matches or beats steering, delete `steer_delta`.
2. **Held-out physical metric for early stopping**, given §2.3(a).
3. **Direct differentiable overhang/thickness loss as the baseline to beat.**
   If a smooth relaxation matches DPO on those two terms, DPO's remit narrows to
   topology alone.
4. **Reflow after DPO**, to recover the straightness DPO costs.

---

## Reproduce

```bash
python trellis_core/flow_dpo_test.py        # 14 tests
python devlab/alignment_experiment.py       # ~104 s, writes concept_data.json
python -m devlab.server                     # dashboard at /concept
```
