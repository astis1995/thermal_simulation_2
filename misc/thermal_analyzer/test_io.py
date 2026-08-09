from pathlib import Path

from config import load_config
from io_utils import (
    find_ir_files,
    load_temperature_csv,
)

# ----------------------------------------------------
# CHANGE THIS
# ----------------------------------------------------

folder = Path("/mnt/c/Documents/thermal_simulation_2/inputs/empiric/Camara termica/Thermal Data 20260717/CICIMAUCR0105-kalinini/images/117-125")
# ----------------------------------------------------

files = find_ir_files(folder)

print(f"Found {len(files)} files\n")

for f in files[:5]:
    print(f)

print()

matrix, metadata = load_temperature_csv(files[0].csv)

print("Matrix shape:")
print(matrix.shape)

print()

print("Temperature range:")
print(matrix.min(), matrix.max())

print()

print("Metadata:")

for k, v in metadata.items():
    print(f"{k}: {v}")
