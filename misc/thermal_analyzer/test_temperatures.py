from pathlib import Path

from io_utils import (
    find_ir_files,
    load_image,
    load_temperature_csv,
)

from point_selector import (
    select_points,
)

from temperature_extractor import (
    load_temperature_series,
)

folder = Path(
    "/mnt/c/Documents/thermal_simulation_2/inputs/empiric/Camara termica/Thermal Data 20260717/CICIMAUCR0105-kalinini/images/117-125"
)

files = find_ir_files(folder)

first = files[0]

image = load_image(first.jpg)

matrix, _ = load_temperature_csv(first.csv)

points = select_points(
    image=image,
    csv_shape=matrix.shape,
    max_points=10,
)

df = load_temperature_series(
    files=files,
    points=points,
    averaging_radius=1,
    timestamp_priority=["ocr"],
)

print(df)

df.to_csv("temperatures.csv", index=False)

print("\nSaved temperatures.csv")
