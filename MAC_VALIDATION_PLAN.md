# Mac Validation Plan — Physics-Aware DPO Pipeline

**Purpose:** everything in `trellis_core/geometric_judge.py`, `dpo_branch.py`,
and `dpo_generation.py` was written against the cloned TRELLIS.2 source and
reviewed by re-reading, but **never executed** — there's no MPS device or
model weights on the machine it was built on. This is the plan for the first
session that actually runs it on your Mac, in order, with a way to tell at
each step whether to keep going, adjust a knob, or that something is
genuinely broken.

Read this top to bottom once before starting — the order matters. Each step
exists to isolate one specific unverified assumption, so if something breaks,
you'll already know which step (and which assumption) it was.

---

## 0. Before touching any new code: confirm the baseline still works

If this step fails, nothing below it is worth trying yet — fix the
environment first.

```bash
cd Solidify
git checkout physics-aware-pipeline
git pull                      # if you pushed from elsewhere, or:
git log --oneline -5          # confirm you see the DPO commits

# if deps/, TRELLIS.2/, .venv/ don't exist yet on this Mac:
bash setup.sh                 # or SKIP_METAL=1 bash setup.sh to go faster first

source .venv/bin/activate

# the existing, unmodified, known-working path -- confirms weights are
# downloaded and the environment itself is healthy before adding anything new
python generate.py assets/shoe_input.png --pipeline-type 512 --steps 12 --output /tmp/baseline_test
```

**Aim for:** a GLB gets written, no crash, roughly matches the timings in
`README.md`'s benchmark table (don't worry if it's slower on a cold cache —
just that it *completes*).

**If this fails:** it's a setup/weights/dependency problem, unrelated to the
DPO work. Fix it the normal way (re-run `setup.sh`, check `hf auth login`,
etc.) before going further.

---

## 1. Run both test suites on the Mac itself

These are CPU-only and platform-independent in principle, but "in principle"
is exactly the kind of claim this whole plan exists to check.

```bash
python trellis_core/geometric_judge_test.py
python trellis_core/dpo_branch_test.py
```

**Aim for:** both print `All N tests passed.` (8 and 13 respectively, as of
this writing).

**If something fails here:** it's almost certainly a dependency gap
(`rtree`, `fast_simplification`) rather than a Mac-specific issue — `pip
install` whatever's missing and re-run. If a test fails for a reason that
looks platform-specific, stop and figure out why before moving to step 2 —
you want these green before trusting anything downstream.

---

## 2. Isolate the highest-risk piece: does the branching hook even run?

This is the step that actually answers "does this theory work at all,"
mechanically speaking — before asking whether it's a *good* idea, first
confirm it *runs*. Write this as a small throwaway script (don't add it to
git yet — call it e.g. `/tmp/smoke_dpo.py` or drop it in
`AppData`/scratch-equivalent on the Mac):

```python
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis_core
from trellis_core.pipeline import load_pipeline
from trellis_core import dpo_branch
from PIL import Image as PILImage

pipeline = load_pipeline("microsoft/TRELLIS.2-4B", device="mps")
img = pipeline.preprocess_image(PILImage.open("assets/shoe_input.png"))

import torch
torch.manual_seed(42)
cond = pipeline.get_cond([img], 512)
coords = pipeline.sample_sparse_structure(cond, 32, 1, {"steps": 12})

flow_model = pipeline.models["shape_slat_flow_model_512"]

t0 = time.time()
shape_slat, report = dpo_branch.sample_shape_slat_with_dpo_branch(
    pipeline, cond, flow_model, coords,
    sampler_params={"steps": 12},
    dpo_config=dpo_branch.DPOBranchConfig(decode_resolution=512, verbose=True),
    return_report=True,
)
print(f"\nDone in {time.time() - t0:.1f}s")
print(report)
```

**What this isolates, specifically** (each of these is a real "UNVERIFIED"
comment in the code — this step is what turns them into "confirmed" or "found
a bug"):
- Does `sampler._get_model_prediction(...)` called directly (bypassing
  `sample()`) actually work against the real `FlowEulerCfgSampler` /
  `FlowEulerGuidanceIntervalSampler` — does `backfill_sampler_defaults`
  actually prevent the `TypeError` it was built to prevent?
- Does `SparseTensor.replace(feats)` behave as read from source?
- Does `pipeline.decode_shape_slat()` on a mid-trajectory `pred_x_0` produce
  a *decodable, non-empty* mesh (not garbage, not a crash)?
- Does the `sparse_conv_backend("none")` override work, or print the
  "could not switch" warning (meaning `patches/mps_compat.py` hasn't run, or
  `conv_none.py` isn't installed in this TRELLIS.2 checkout)?
- Does anything OOM or trip the macOS GPU watchdog partway through?

**Aim for:** it completes and prints a `DPOBranchReport`. Read
`report.notes` — if it contains "a candidate failed to decode," that's not
a crash, but it does mean this run's steering did nothing (see step 4).

**If it crashes:** the traceback tells you which of the five things above
broke. Common ones to expect, from most to least likely:
- `TypeError: ... missing required positional argument` → the
  `backfill_sampler_defaults` fix didn't cover something; check what argument
  it's complaining about and whether it's in `sig.parameters` at all (maybe
  the real sampler class has a different signature than the version of
  `flow_euler.py` this was written against — check with
  `print(type(pipeline.shape_slat_sampler).__mro__)` and
  `inspect.signature(type(pipeline.shape_slat_sampler).sample)`).
- A Metal/MPS error mentioning `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
  → the GPU watchdog (see `trellis_core/pipeline.py`'s `WatchdogError` /
  README's documented workarounds: run headless, or
  `MTL_CAPTURE_ENABLED=1`).
- Out of memory → reduce `continuation_steps` and/or `num_delta_grad_steps`
  in `DPOBranchConfig` before anything else.

---

## 3. Run the full pipeline end-to-end

Once step 2 works, try the actual Phase-4-wired path:

```python
from trellis_core import dpo_generation

result = dpo_generation.run_generation_with_dpo(
    pipeline,
    PILImage.open("assets/shoe_input.png"),
    seed=42,
    pipeline_type="512",
    steps=12,
    output_prefix="/tmp/dpo_test/shoe",
)
print(result)
print(result.printable_result.diagnostics)
print(result.dpo_report)
```

**Aim for:** a `.glb`/`.stl` pair written to `/tmp/dpo_test/`, and a
`PrintableResult` with `watertight=True` (or at least not wildly different
from what the vanilla path produces on the same image — see step 4).

---

## 4. The actual experiment: does the steering do anything measurable?

This is the part that answers "can this theory work," not just "does it
run." The independent review that went over this code flagged a specific,
concrete concern: the differentiable gradient signal (a latent detail-energy
proxy) and what the judge actually rewards (dominated by the topology term)
might be uncorrelated — meaning the branch step could reduce to "smarter
proposal, kept only if the judge likes it after the fact" rather than real
steering. This step is how you find out which one it actually is, empirically,
instead of arguing about it.

**Do this for at least 5-10 different input images** (more is better; single
runs are not evidence of anything given how much noise is in a single
generation):

1. **Generate the vanilla baseline** for each image with
   `trellis_core.pipeline.run_generation` (pipeline_type `512`, same seed).
2. **Generate the DPO version** for each image with
   `dpo_generation.run_generation_with_dpo` (same seed, same pipeline_type).
3. **For every run, record:**
   - `dpo_report.delta_branch_won_initial` and `.delta_branch_won_final` —
     tally these across all runs. **This is the single most important number
     from this whole session.** If the optimized (`_final`) branch wins
     against the reference notably more often than the un-optimized initial
     candidate did, the gradient step is contributing something real. If the
     win rates are statistically indistinguishable, the steering step isn't
     doing anything beyond what a random perturbation + re-ranking would do
     for free — that's a genuine, useful negative result, not a failure of
     this validation session.
   - `printable_result.diagnostics` (overhang %, thin-wall warning count,
     watertight) for both the vanilla and DPO output on the same image —
     compare them side by side.
   - `printable_result.fidelity` (chamfer/hausdorff/volume-change vs the
     pre-repair mesh) for both — a sanity check that DPO output isn't
     *decisively worse* going into repair.
   - Wall-clock time for both (`generation_seconds` on each result) — this is
     the real cost of the branch step on your actual hardware, not the
     theoretical estimate in the plan doc's Phase 2 docstring.
   - Peak memory during the DPO run (Activity Monitor, or `/usr/bin/time -l`
     around the script if you want a number) — the plan's #1 flagged risk was
     unified-memory pressure during branching; this tells you if it's real.

4. **Look at the meshes themselves.** Open a handful of the DPO outputs next
   to their vanilla counterparts in a viewer. Does the branched one look
   different at all? Garbled? Indistinguishable? This matters as much as the
   numbers — a steering effect too small to see is a steering effect too
   small to matter yet.

**What "this theory can work" looks like after this step:** the optimized
branch wins measurably more than the initial random one across your sample,
*and* the DPO output's printability diagnostics are as good or better than
vanilla on average, *and* the overhead is tolerable on your hardware. All
three need to hold — a steering effect that wins more often but produces
worse prints, or one that's real but costs 10x the generation time, both need
more work before this is useful, not a green light.

**What "needs more work, but the idea survives" looks like:** the win-rate
signal is weak or absent, but nothing crashed and the overhead is acceptable.
In that case the next thing to try (not today, but next session) is
adjusting `JudgeWeights` (`gamma`/`delta` especially — the review flagged
`L_Topo` as likely still dominating even after being turned into a rate) or
`DPOBranchConfig.dpo_beta`/`branch_noise_scale`, and re-running this same
measurement, rather than declaring it broken from one day's data.

**What "doesn't work as designed" looks like:** the optimized branch's win
rate is not distinguishable from chance across a real sample, even after
trying a couple of weight adjustments. That's a legitimate outcome for a
Phase-2/3 research scaffold — it would mean the latent detail-energy proxy
genuinely isn't a usable steering signal for this judge, and the next design
iteration should look at whether a different differentiable proxy (or a
different branch-selection strategy entirely) is worth trying, per the
"HONEST LIMITATIONS" section already written into `dpo_branch.py`.

---

## 5. Explicitly out of scope for tomorrow

Don't do these yet, even if step 4 goes well — they're follow-up work once
the DPO step is validated on its own:

- Wiring this into `generate.py` or `server/` (deliberately deferred; the
  interface may need to change based on what you learn in step 4).
- The CUDA track (Track B) — no CUDA hardware exists to test it on anyway.
- Adapting for `pipeline_type='1024_cascade'` (the repo's actual default) —
  `dpo_branch.py` only supports `'512'`/`'1024'` right now; that's a
  documented, deliberate gap, not something to patch today.
- Tuning `JudgeWeights`/`DPOBranchConfig` extensively before you have the
  step-4 baseline numbers to tune against — you'd be guessing blind.

---

## Quick reference: what to watch for at each step

| Step | What could go wrong | Where to look |
|---|---|---|
| 0 | Weights/deps missing | `setup.sh` output, `hf auth login` |
| 1 | Missing `rtree`/`fast_simplification` | `pip install` them |
| 2 | Missing guidance kwarg, SparseTensor API mismatch, OOM, watchdog | traceback; `dpo_branch.py`'s `UNVERIFIED` comments point at the exact assumption that broke |
| 3 | `printable_result.watertight=False`, crash in post-processing | `trellis_core/printable.py`'s existing (unmodified, already-trusted) diagnostics |
| 4 | Win rate indistinguishable from chance, worse diagnostics than vanilla, unacceptable overhead | `dpo_report` fields, side-by-side `diagnostics`/`fidelity`, wall-clock, Activity Monitor |
