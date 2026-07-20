# stl_bounds.py

import sys
import numpy as np
import trimesh


def analyze_stl(stl_path):
    mesh = trimesh.load_mesh(stl_path)

    vertices = mesh.vertices

    xmin_idx = np.argmin(vertices[:, 0])
    xmax_idx = np.argmax(vertices[:, 0])

    ymin_idx = np.argmin(vertices[:, 1])
    ymax_idx = np.argmax(vertices[:, 1])

    zmin_idx = np.argmin(vertices[:, 2])
    zmax_idx = np.argmax(vertices[:, 2])

    xmin_pt = vertices[xmin_idx]
    xmax_pt = vertices[xmax_idx]

    ymin_pt = vertices[ymin_idx]
    ymax_pt = vertices[ymax_idx]

    zmin_pt = vertices[zmin_idx]
    zmax_pt = vertices[zmax_idx]

    print("\n==============================")
    print("STL GEOMETRY ANALYSIS")
    print("==============================")

    print("\nXYZ Limits")
    print(f"X: {vertices[:,0].min():.6f} -> {vertices[:,0].max():.6f}")
    print(f"Y: {vertices[:,1].min():.6f} -> {vertices[:,1].max():.6f}")
    print(f"Z: {vertices[:,2].min():.6f} -> {vertices[:,2].max():.6f}")

    print("\nBounding Box Size")
    print(f"ΔX = {np.ptp(vertices[:,0]):.6f}")
    print(f"ΔY = {np.ptp(vertices[:,1]):.6f}")
    print(f"ΔZ = {np.ptp(vertices[:,2]):.6f}")

    print("\nExtreme Points")

    print("\nX MIN")
    print(f"  ({xmin_pt[0]:.6f}, {xmin_pt[1]:.6f}, {xmin_pt[2]:.6f})")

    print("\nX MAX")
    print(f"  ({xmax_pt[0]:.6f}, {xmax_pt[1]:.6f}, {xmax_pt[2]:.6f})")

    print("\nY MIN")
    print(f"  ({ymin_pt[0]:.6f}, {ymin_pt[1]:.6f}, {ymin_pt[2]:.6f})")

    print("\nY MAX")
    print(f"  ({ymax_pt[0]:.6f}, {ymax_pt[1]:.6f}, {ymax_pt[2]:.6f})")

    print("\nZ MIN")
    print(f"  ({zmin_pt[0]:.6f}, {zmin_pt[1]:.6f}, {zmin_pt[2]:.6f})")

    print("\nZ MAX")
    print(f"  ({zmax_pt[0]:.6f}, {zmax_pt[1]:.6f}, {zmax_pt[2]:.6f})")

    print("\nCenter of Bounding Box")

    center = (
        vertices.min(axis=0) +
        vertices.max(axis=0)
    ) / 2

    print(
        f"({center[0]:.6f}, "
        f"{center[1]:.6f}, "
        f"{center[2]:.6f})"
    )

    print("\nMesh Info")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces   : {len(mesh.faces)}")

    try:
        print(f"Watertight: {mesh.is_watertight}")
    except Exception:
        pass


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python stl_bounds.py path/to/model.stl")
        sys.exit(1)

    analyze_stl(sys.argv[1])
