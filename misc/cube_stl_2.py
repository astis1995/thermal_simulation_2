import numpy as np
from stl import mesh

# Cube dimensions
L = 10.0
W = 10.0
H = 10.0

# Vertices
vertices = np.array([
    [0, 0, 0],      # 0
    [L, 0, 0],      # 1
    [L, W, 0],      # 2
    [0, W, 0],      # 3
    [0, 0, H],      # 4
    [L, 0, H],      # 5
    [L, W, H],      # 6
    [0, W, H],      # 7
])

# Two triangles per face (12 total)
faces = np.array([
    [0, 3, 1], [1, 3, 2],   # Bottom
    [4, 5, 7], [5, 6, 7],   # Top
    [0, 1, 4], [1, 5, 4],   # Front
    [1, 2, 5], [2, 6, 5],   # Right
    [2, 3, 6], [3, 7, 6],   # Back
    [3, 0, 7], [0, 4, 7],   # Left
])

# Create mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for i, face in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[face[j]]

cube.save("cube_10x10x10.stl")

print("Created cube_10x10x10.stl")
