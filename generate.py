"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis_core  # noqa: E402  (must be first trellis-related import; sets up env/sys.path)
from trellis_core.pipeline import load_pipeline, run_generation, WatchdogError, watchdog_help_message  # noqa: E402

import argparse
import time
from PIL import Image as PILImage


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="output_3d", help="Output filename without extension (default: output_3d)")
    parser.add_argument(
        "--pipeline-type", default="1024_cascade",
        choices=["512", "1024", "1024_cascade"],
        help="Pipeline resolution (default: 1024_cascade)",
    )
    parser.add_argument(
        "--texture-size", type=int, default=2048,
        choices=[512, 1024, 2048],
        help="Texture resolution for PBR baking (default: 2048)",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip texture baking, export geometry only",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override sampler steps for all three flow phases (default: pipeline JSON, usually 12)",
    )
    parser.add_argument(
        "--target-faces", type=int, default=1000000,
        help="Pre-bake simplification target face count (default: 1000000)",
    )
    parser.add_argument(
        "--model-id", default="microsoft/TRELLIS.2-4B",
        help="HuggingFace model id to load (default: microsoft/TRELLIS.2-4B)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon")
    print("=" * 60)

    print("\nLoading pipeline...")
    t0 = time.time()
    pipeline = load_pipeline(args.model_id, device="mps")
    print(f"Loaded in {time.time() - t0:.0f}s")
    print("Device: MPS")

    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")

    print(f"\nGenerating 3D model (pipeline={args.pipeline_type}, seed={args.seed})...")

    glb_path = f"{args.output}.glb"
    obj_path = f"{args.output}.obj"

    try:
        result = run_generation(
            pipeline,
            img,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            steps=args.steps,
            target_faces=args.target_faces,
            texture_size=args.texture_size,
            no_texture=args.no_texture,
            out_glb_path=glb_path,
            out_obj_path=obj_path,
        )
    except WatchdogError as e:
        print(f"\nERROR: {e}")
        sys.exit(2)

    print(f"\nMesh: {result.vertex_count:,} vertices, {result.face_count:,} triangles")
    print(f"Generation time: {result.generation_seconds:.1f}s")
    if result.used_metal_bake:
        print(f"\nBaked PBR textures via Metal ({args.texture_size}x{args.texture_size})...")
    elif not args.no_texture:
        print(f"\nBaked PBR textures via KDTree ({args.texture_size}x{args.texture_size})...")
    else:
        print("\nExported with vertex colors (use without --no-texture for PBR textures)...")
    print(f"  Bake time: {result.bake_seconds:.0f}s")
    print(f"Saved: {result.glb_path}")
    print(f"Saved: {result.obj_path}")

    print(f"\nTotal time: {result.generation_seconds:.1f}s generation + baking")


if __name__ == "__main__":
    main()
