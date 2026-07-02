import trimesh
import sys
import os

def convert(input_path):
    # Load the mesh
    mesh = trimesh.load(input_mesh)

    # Generate the output filename
    output_path = os.path.splitext(input_path)[0] + ".stl"

    # Export to STL
    mesh.export(output_path)
    print(f"Success! Converted {input_path} to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <path_to_obj>")
    else:
        input_mesh = sys.argv[1]
        convert(input_mesh)

