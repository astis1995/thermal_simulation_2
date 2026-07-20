from pathlib import Path

import trimesh
import pymeshfix

# --------------------------------------------------------
# Load STL
# --------------------------------------------------------

input_stl = Path(
    "/mnt/c/Documents/thermal_simulation_2/inputs/simulation1/stl/elitro_limpio_blender3_fixed_mm.stl"
)

output_stl = input_stl.with_name(
    input_stl.stem + "_meshfix.stl"
)

mesh = trimesh.load_mesh(input_stl)

print("=" * 60)
print("BEFORE")
print("=" * 60)
print("Watertight :", mesh.is_watertight)
print("Volume     :", mesh.is_volume)
print("Consistent :", mesh.is_winding_consistent)
print("Euler      :", mesh.euler_number)

# --------------------------------------------------------
# Repair
# --------------------------------------------------------

repair = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
repair.repair()

fixed = trimesh.Trimesh(
    vertices=repair.points,
    faces=repair.faces,
    process=False,
)

print("\n" + "=" * 60)
print("AFTER")
print("=" * 60)
print("Watertight :", fixed.is_watertight)
print("Volume     :", fixed.is_volume)
print("Consistent :", fixed.is_winding_consistent)
print("Euler      :", fixed.euler_number)

fixed.export(output_stl)

print("\nSaved:", output_stl)
