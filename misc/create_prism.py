#!/usr/bin/env python3

import sys
import numpy as np
from stl import mesh


def create_prism(x, y, z):
    """
    Create a rectangular prism STL.

    Parameters
    ----------
    x : float
        Length in X (mm)

    y : float
        Length in Y (mm)

    z : float
        Length in Z (mm)
    """

    vertices = np.array([
        [0, 0, 0],      # 0
        [x, 0, 0],      # 1
        [x, y, 0],      # 2
        [0, y, 0],      # 3
        [0, 0, z],      # 4
        [x, 0, z],      # 5
        [x, y, z],      # 6
        [0, y, z],      # 7
    ], dtype=float)

    faces = np.array([
        [0, 3, 1], [1, 3, 2],   # Bottom
        [4, 5, 7], [5, 6, 7],   # Top
        [0, 1, 4], [1, 5, 4],   # Front
        [1, 2, 5], [2, 6, 5],   # Right
        [2, 3, 6], [3, 7, 6],   # Back
        [3, 0, 7], [0, 4, 7],   # Left
    ])

    prism = mesh.Mesh(
        np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    )

    for i, face in enumerate(faces):
        for j in range(3):
            prism.vectors[i][j] = vertices[face[j]]

    filename = f"prism_{x:g}_{y:g}_{z:g}.stl"

    prism.save(filename)

    print(f"Created {filename}")
    print(f"Dimensions: {x} × {y} × {z} mm")


def main():

    if len(sys.argv) != 4:
        print("Usage:")
        print("    python create_prism.py X Y Z")
        print("")
        print("Example:")
        print("    python create_prism.py 10 20 30")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    z = float(sys.argv[3])

    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("All dimensions must be positive.")

    create_prism(x, y, z)


if __name__ == "__main__":
    main()
