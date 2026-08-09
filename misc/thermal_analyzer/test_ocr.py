from pathlib import Path
from io_utils import _timestamp_from_ocr

jpg = Path(
    "/mnt/c/Documents/thermal_simulation_2/inputs/empiric/Camara termica/Thermal Data 20260717/CICIMAUCR0105-kalinini/images/117-125/IR_00117.jpg"
)

ts = _timestamp_from_ocr(jpg)

print(ts)
