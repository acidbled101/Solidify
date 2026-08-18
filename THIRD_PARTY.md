# What in this repository is not mine

Everything outside `third_party/` is my own work. This page exists so you do
not have to take that on faith.

## Verify it in one command

```bash
git diff --stat d58628f..HEAD
```

`d58628f` (2026-04-28) is the last commit of the Apple Silicon port this
project started from. Everything after it is mine: **267 files changed, ~111k
insertions** — or **~28,000 lines across 120 files** once vendored JavaScript,
test photographs and rendered PDFs are excluded.

I did not modify the upstream code. All four of its backends are still
byte-for-byte what they were:

```bash
git diff -M d58628f..HEAD -- backends patches third_party
```

| file | status |
|---|---|
| `third_party/backends/conv_none.py` | **byte-identical**, moved only |
| `third_party/backends/mesh_extract.py` | **byte-identical**, moved only |
| `third_party/backends/stubs.py` | **byte-identical**, moved only |
| `third_party/backends/texture_baker.py` | **byte-identical**, moved only |
| `third_party/patches/mps_compat.py` | moved; **+17/−5, path constants only** |

The one exception is honest and small: `mps_compat.py` computed its paths as
`dirname(dirname(__file__))`, which broke when the file moved one directory
deeper. Only those constants changed — nothing about what the patches *do*.
Everything else that could have been rewritten was not: `bootstrap.py` puts
`third_party/` on `sys.path` instead, so `import backends.texture_baker`
resolves exactly as upstream wrote it.

## The three layers

**Microsoft TRELLIS.2 — the model.** Not in this repository at all. `setup.sh`
clones it into `TRELLIS.2/`, which is gitignored. I use it unmodified except
for the MPS compatibility patches in `third_party/patches/`, which are also not
mine. MIT licensed.

**shivampkumar/trellis-mac — the Apple Silicon port.** `third_party/backends/`
(4 files) and `third_party/patches/` (1 file), MIT licensed, by Shivam Kumar
with contributions from Xiang Li. Replaces TRELLIS.2's CUDA-only dependencies
— `flex_gemm`, `o_voxel`'s hashmap, `flash_attn`, `nvdiffrast` — with Metal and
pure-PyTorch equivalents so the model runs on Apple Silicon. Its original
documentation is preserved verbatim at `third_party/README.md`. The Metal
libraries it depends on (`mtlgemm`, `mtldiffrast`, `mtlbvh`, `mtlmesh`) are by
[@pedronaugusto](https://github.com/pedronaugusto) and are installed by
`setup.sh`, not vendored here.

**This project.** Everything else.

## Other third-party content

| what | where | licence |
|---|---|---|
| three.js, GLTFLoader, OrbitControls, BufferGeometryUtils | `experiments/dpo_inference_steering/inspector/static/vendor/` | MIT |
| The "Solidify" frontend visual design | `server/static/` | by Ajlan AlAjlan, 4 commits |
| Example shoe renders (upstream's) | `third_party/assets/` | Shivam Kumar |
| Thingi10K source models | not vendored — see `data/thingi10k_sft/ATTRIBUTION.md` | per-object, mixed CC |
| DINOv3 | downloaded at setup | Meta custom licence (gated) |
| RMBG-2.0 | downloaded at setup | CC BY-NC 4.0 (non-commercial) |

## Contributors

`git shortlog -sn` on this branch, with `.mailmap` consolidating the three
identities I committed under across three machines:

```
    52  Ali Alhulaimi
    21  Shivam Kumar      (the Apple Silicon port, Apr 2026)
     4  Ajlan AlAjlan     (Solidify frontend design)
     3  Xiang Li          (setup dependency pre-cloning)
```
