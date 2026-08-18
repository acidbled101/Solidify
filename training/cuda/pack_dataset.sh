#!/bin/zsh
# Package the 5,020-object Thingi10K SLat dataset for transfer to the GPU box.
#
#   ./pack_dataset.sh [DATA_DIR] [OUT_DIR]
#
# Produces two tarballs, because they are wildly different sizes and only one
# of them is optional:
#
#   thingi10k_5k_core.tar   ~725 MB   latents/ + manifest.jsonl   REQUIRED
#   thingi10k_5k_cond.tar   ~9.9 GB   cond/                       REQUIRED
#   thingi10k_5k_images.tar ~300 MB   images/                     not used by
#                                                                 training
#
# `cond/` is 93% of the bytes: one DINOv3 feature grid per object, [1,1029,1024]
# fp16 = 2.11 MB each. It is precomputed rather than recomputed on the GPU box
# on purpose -- recomputing it would drag in DINOv3, BiRefNet background
# removal and the renderer, which is most of the dependency surface this
# package exists to avoid.
#
# No gzip: .npz is already zip-compressed and the fp16 .npy files barely
# compress, so gzip would cost ~20 minutes of CPU to save a few percent.

set -e
DATA=${1:-data/thingi10k_3k}
OUT=${2:-dist}

if [ ! -f "$DATA/manifest.jsonl" ]; then
  echo "no manifest at $DATA/manifest.jsonl" >&2; exit 1
fi

mkdir -p "$OUT"
N=$(wc -l < "$DATA/manifest.jsonl" | tr -d ' ')
echo "packing $N objects from $DATA -> $OUT"

tar -cf "$OUT/thingi10k_5k_core.tar"   -C "$DATA" manifest.jsonl latents
tar -cf "$OUT/thingi10k_5k_cond.tar"   -C "$DATA" cond
tar -cf "$OUT/thingi10k_5k_images.tar" -C "$DATA" images

( cd "$OUT" && shasum -a 256 thingi10k_5k_*.tar > thingi10k_5k.sha256 )

echo
ls -lh "$OUT"/thingi10k_5k_*.tar
echo
cat <<'EOF'
Transfer options, fastest first:

  1. Hugging Face Hub (recommended -- resumable, and the GPU box pulls it at
     line rate instead of you uploading twice):

       hf auth login
       hf upload <your-user>/thingi10k-slat-5k dist/ . --repo-type=dataset --private

     then on the A100 box:

       hf download <your-user>/thingi10k-slat-5k --repo-type=dataset --local-dir data_tar
       for f in data_tar/*.tar; do tar -xf "$f" -C data/thingi10k_5k; done

  2. Direct copy:

       rsync -avP dist/thingi10k_5k_*.tar user@gpubox:/workspace/

Verify after transfer:  shasum -a 256 -c thingi10k_5k.sha256
EOF
