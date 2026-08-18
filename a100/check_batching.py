"""Correctness gate: does a mini-batch give the same loss as one-at-a-time?

WHY THIS EXISTS
---------------
trellis2's sparse attention has two implementations and they are not
equivalent for batches of mixed-length objects.

  flash_attn : flash_attn_varlen_* with cu_seqlens. Objects are packed with no
               padding and each attends only to itself. Correct for any batch.
  sdpa       : pads every object out to the longest in the batch with ZEROS,
               then calls scaled_dot_product_attention with NO attn_mask
               (trellis2/modules/sparse/attention/full_attn.py, the
               `config.ATTN in ('sdpa','naive')` branch). Padded key positions
               take part in the softmax of every real query, so each object's
               output depends on whatever else happened to share its batch.

Nothing raises. There is no NaN. The loss is simply a different number, and
training proceeds happily on a corrupted gradient. The only way to catch it is
to check the invariant directly, which is what this does: run k objects as one
batch and as k batches of one, with identical timesteps and identical noise,
and compare.

Run this ONCE on the training box before committing hours of GPU time.

    python check_batching.py --data <dataset> --trellis-path <TRELLIS.2> --device cuda

Expected: PASS on flash_attn, FAIL on sdpa. A FAIL is not a bug in this script.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_cuda  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--trellis-path", default=os.environ.get("TRELLIS2_PATH", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-backend", default=None)
    ap.add_argument("--k", type=int, default=3, help="objects per batch")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="relative tolerance; bf16 alone lands near 1e-3")
    args = ap.parse_args(argv)

    if args.trellis_path and args.trellis_path not in sys.path:
        sys.path.insert(0, args.trellis_path)
    backend = sft_cuda.configure_backend(args.device, args.attn_backend)
    print(f"attention backend: {backend}")

    manifest = sft_cuda.read_manifest(args.data)
    # Deliberately pick objects of DIFFERENT sizes -- equal-length objects are
    # the one case where the padded path happens to be correct, so a batch of
    # them would pass and prove nothing.
    manifest.sort(key=lambda r: r[1])
    picks = [manifest[len(manifest) // 4], manifest[len(manifest) // 2],
             manifest[3 * len(manifest) // 4]][:args.k]
    print("objects: " + ", ".join(f"{fid} ({n} tokens)" for fid, n in picks))

    ds = sft_cuda.SlatDataset(args.data, [fid for fid, _ in picks])
    model, sigma_min = sft_cuda.load_flow_model(args.device)
    model.eval()

    samples = [sft_cuda.to_device(ds[i], args.device) for i in range(len(ds))]
    k = len(samples)
    torch.manual_seed(0)
    t = torch.rand(k, device=args.device).clamp(1e-3, 1.0)
    eps = [torch.randn(s[1].shape, device=args.device) for s in samples]

    with torch.no_grad():
        batched, _ = sft_cuda.flow_matching_loss(
            model, samples, sigma_min, p_uncond=0.0, t_override=t, eps_override=eps)
        singles = [
            float(sft_cuda.flow_matching_loss(
                model, [samples[i]], sigma_min, p_uncond=0.0,
                t_override=t[i:i + 1], eps_override=[eps[i]])[0])
            for i in range(k)]

    ref = sum(singles) / k          # the batch loss is the mean of per-object means
    got = float(batched)
    rel = abs(got - ref) / max(abs(ref), 1e-12)
    print(f"\none at a time : {['%.6f' % s for s in singles]}  mean {ref:.6f}")
    print(f"as one batch  : {got:.6f}")
    print(f"relative difference: {rel:.3%}  (tolerance {args.tol:.3%})")

    if rel <= args.tol:
        print("\nPASS -- mini-batching is equivalent. --max-batch > 1 is safe.")
        return 0
    print("\nFAIL -- batching changes the loss. Do NOT train with --max-batch > 1 on\n"
          f"this backend ({backend}); install flash-attn, or use --max-batch 1.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
