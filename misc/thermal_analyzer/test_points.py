from pathlib import Path

from io_utils import (
    find_ir_files,
    load_image,
    load_temperature_csv,
)

from point_selector import (
    select_points,
    save_selected_points_image,
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

print("\nSelected points:")

for p in points:
    print(p)

save_selected_points_image(
    image=image,
    points=points,
    csv_shape=matrix.shape,
    output_file="selected_points.png",
)

print("\nImage saved as selected_points.png")
