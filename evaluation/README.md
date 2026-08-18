# Evaluation

Measurement harnesses. Nothing here runs in production; everything here is how
Solidify's claims were checked.

| file | what it measures |
|---|---|
| `topology_test.py` | mesh topology: non-manifold edge rate, open edge rate, connected components, detail (surface energy). The metrics behind the −44% figure. |
| `compare_models.py` | runs the same photo through several checkpoints and exports the raw meshes for comparison |
| `geometric_judge.py` | physics-aware mesh scoring — overhang, thickness, topology, detail |
| `compare_printable.py` | A/B harness for the print-prep pipelines (v1 / v2 / v3) |
| `verify_adapter.py` | that the LoRA loads, wraps 210 modules, and that disabling it is bit-exact |
| `bench_preview.py` | cost of decoding an in-flight preview |
| `test_progress_hooks.py` | that the sampler progress hooks fire |

Tests are plain-assert with a `__main__` runner — no pytest in this repo:

```bash
python -m evaluation.geometric_judge_test
```

## Never measure topology on a textured export

Every figure in this repository is measured on **geometry-only** output. Run
the same mesh through the texture bake and the numbers become meaningless,
because xatlas splits vertices along every UV seam -- the surface still looks
continuous, but topologically it has been shredded.

Measured on one image through `sft-1200@1050`, the same model and checkpoint
in both rows:

| | faces | open edges | components |
|---|---:|---:|---:|
| geometry-only | 1,476,706 | 0.804% | 3,639 |
| after texture bake | 499,999 | **46.767%** | **68,376** |

A 58x difference in open-edge rate, from a step that changes nothing about the
shape. So: `generate.py --no-texture`, or `TRELLIS_NO_TEXTURE=1` (which is
already the server's default), before `topology_test.py` means anything.

## A caveat about the comparison set

`runs/comparison_900/INDEX.md` describes its inputs as "5 objects". They are
**two**: hashing them shows a 3DBenchy used twice (byte-identical) and one
portrait used three times (two byte-identical, one JPEG re-encode). Different
trace directories disguised it.

The tuned model wins on both subjects and the per-image tables are correct, but
any claim of the form "measured across five test images" is an overclaim, and
the averages are weighted 3:2 toward the portrait. `test/` holds ~30 unrelated
photos and `compare_models.py` takes an input list, so a clean re-run is one
batch.
