# Licences of the models Solidify depends on

Solidify's own code is MIT (see [`../LICENSE`](../LICENSE)). The models it runs
are **not** — two of them restrict what you may do with the output, and both
are gated, so you must accept their terms on Hugging Face before `setup.sh`
can download anything.

| model | role | licence | commercial use |
|---|---|---|---|
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | image → 3D | MIT | yes |
| [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | image features | Meta DINOv3 Licence — [full text](DINOv3-LICENSE.md) | see the licence; it is a custom Meta agreement, not open source |
| [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | background removal | BRIA declares `bria-rmbg-2.0`, pointing at CC BY-NC 4.0 — [full text](CC-BY-NC-4.0.txt) | **no** — requires a separate licence from BRIA |

## What this means in practice

**RMBG-2.0 is the binding constraint.** It is non-commercial, it runs on every
single job, and there is no code path around it — background removal happens
before the model sees the image. So a commercial deployment of Solidify needs a
commercial licence from [BRIA](https://bria.ai/), regardless of anything in
this repository's own MIT licence.

**DINOv3 is a custom Meta agreement**, not a standard open-source licence. Read
[`DINOv3-LICENSE.md`](DINOv3-LICENSE.md) before deploying; it carries its own
acceptable-use and attribution conditions.

**The training data has its own terms again** — the 300 Thingi10K models carry
per-object Creative Commons licences, 11 of them No-Derivatives and 94
non-commercial. See
[`../data/thingi10k_sft/ATTRIBUTION.md`](../data/thingi10k_sft/ATTRIBUTION.md).

## Provenance of these files

`DINOv3-LICENSE.md` is the file Meta ships in the model repository, copied
verbatim. `CC-BY-NC-4.0.txt` is the canonical Creative Commons legal code from
creativecommons.org — BRIA ships no LICENSE file of its own, so this is the
text its model card points to. Neither is my writing, and neither is a summary:
the table above is a convenience, and the licence files govern.
