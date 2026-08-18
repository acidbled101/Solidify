"""Convert a mesh to STL.

    python -m tools.convert model.glb        # -> model.stl
"""

import os
import sys

import trimesh


def convert(input_path):
    # This read `input_mesh` -- a global that only existed because __main__
    # happened to assign it -- rather than its own argument, so calling
    # convert() from anywhere else raised NameError or silently converted
    # whatever the last CLI invocation had loaded.
    mesh = trimesh.load(input_path)

    output_path = os.path.splitext(input_path)[0] + ".stl"
    mesh.export(output_path)
    print(f"Converted {input_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.convert <mesh>")
        sys.exit(1)
    convert(sys.argv[1])
