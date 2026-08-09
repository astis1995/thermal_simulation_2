from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

folder = Path(r"/mnt/c/Documents/thermal_simulation_2/inputs/empiric/Camara termica/Thermal Data 20260717/CICIMAUCR0105-kalinini/images/79-116")

# Radius of averaging window
window_radius = 3      # 5x5 average

# (x,y) = (column,row)
points = [
    (56,46),
    (60,46),
    (65,46),
]

# ---------------------------------------------------------------------

timestamp_pattern = re.compile(
    r"IR_\d+_(\d+-\d+-\d+_\d+\.\d+\.\d+)",
    re.IGNORECASE,
)

def average_temperature(frame, x, y, radius):
    """
    Compute the average temperature around (x,y).

    Parameters
    ----------
    frame : ndarray
    x, y : int
        Pixel coordinates (numpy indexing)
    radius : int
        Radius of square neighborhood.

    Returns
    -------
    float
    """

    h, w = frame.shape

    xmin = max(0, x - radius)
    xmax = min(w, x + radius + 1)

    ymin = max(0, y - radius)
    ymax = min(h, y + radius + 1)

    window = frame[ymin:ymax, xmin:xmax]

    return np.nanmean(window)

def parse_timestamp(filename):
    m = timestamp_pattern.search(filename)
    if m is None:
        raise ValueError(f"Cannot parse timestamp from {filename}")

    return datetime.strptime(m.group(1), "%m-%d-%y_%H.%M.%S")


def heating_model(t, Tmin, Trise, tau):
    return Tmin + Trise * (1 - np.exp(-t / tau))


# ---------------------------------------------------------------------

files = sorted(folder.glob("*.csv"))

times = []
frames = []

for f in files:

    t = parse_timestamp(f.name)

    # first row contains column numbers
    # first column contains row numbers
    df = pd.read_csv(
        f,
        encoding="utf-16",
        skiprows=3,
        header=0,
        index_col=0,
    )

    # Convert everything to numbers
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Remove empty rows (if any)
    df = df.dropna(axis=0, how="all")

    frame = df.to_numpy(dtype=float)

    print(df.columns)
    print(df.index)
    frames.append(df.values.astype(float))
    times.append(t)
    print("Frame shape:", frames[0].shape)

times = np.array([(t - times[0]).total_seconds() for t in times])

print(f"{len(frames)} frames loaded.")

# ---------------------------------------------------------------------

for x, y in points:

    temperatures = np.array([
        average_temperature(frame, x, y, window_radius)
        for frame in frames
    ])

    if np.any(np.isnan(temperatures)):
        print(f"NaNs at pixel ({x},{y})")
        print(np.where(np.isnan(temperatures))[0])
        print(temperatures)
        continue

    if np.any(np.isinf(temperatures)):
        print(f"Infs at pixel ({x},{y})")
        continue
    Tmin_guess = temperatures.min()
    Trise_guess = temperatures.max() - temperatures.min()
    tau_guess = max(times[-1] / 3, 1)

    try:

        popt, pcov = curve_fit(
            heating_model,
            times,
            temperatures,
            p0=[Tmin_guess, Trise_guess, tau_guess],
            bounds=(
                [-100, 0, 1e-6],      # Tmin, Trise, tau
                [300, 100, np.inf]
            )
        )

        Tmin, Trise, tau = popt

        fit = heating_model(times, *popt)

        print(
            f"Center ({x},{y})"
            f"  window={2*window_radius+1}x{2*window_radius+1}"
            f"  Tmin={Tmin:.3f}"
            f"  Trise={Trise:.3f}"
            f"  tau={tau:.3f} s"
        )

    except RuntimeError:

        print(f"Fit failed for pixel ({x},{y})")
        continue

    plt.figure(figsize=(7, 5))

    plt.plot(
        times,
        temperatures,
        "o",
        label="Measured",
    )

    plt.plot(
        times,
        fit,
        "-",
        linewidth=2,
        label=f"Fit (τ={tau:.2f} s)",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Pixel ({x},{y})")

    plt.grid(True)
    plt.legend()

plt.show()
