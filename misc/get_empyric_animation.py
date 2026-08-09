from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tempfile
import numpy as np

import re
import pandas as pd
from datetime import datetime

from pathlib import Path
from datetime import datetime
import re
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio


def save_thermal_gif(
    folder,
    outfile=None,
    points=None,
    fps=5,
    tick_step=10,
    dpi=150,
):
    """
    Create a GIF from Fluke CSV files.

    Parameters
    ----------
    folder : str or Path
        Folder containing the CSV files.

    outfile : str or Path or None
        Output GIF. If None, saves thermal.gif inside folder.

    points : list[(x,y)] or None
        Pixel coordinates to mark.

    fps : float

    tick_step : int

    dpi : int
    """

    folder = Path(folder)

    if outfile is None:
        outfile = folder / "thermal.gif"
    else:
        outfile = Path(outfile)

    timestamp_pattern = re.compile(
        r"IR_\d+_(\d+-\d+-\d+_\d+\.\d+\.\d+)",
        re.IGNORECASE,
    )

    def parse_timestamp(filename):
        m = timestamp_pattern.search(filename)
        if m is None:
            raise ValueError(f"Cannot parse timestamp from {filename}")
        return datetime.strptime(
            m.group(1),
            "%m-%d-%y_%H.%M.%S",
        )

    csv_files = sorted(folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = []
    timestamps = []

    for f in csv_files:

        df = pd.read_csv(
            f,
            encoding="utf-16",
            skiprows=3,
            header=0,
            index_col=0,
        )

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        frames.append(df.to_numpy(dtype=float))
        timestamps.append(parse_timestamp(f.name))

    times = np.array(
        [
            (t - timestamps[0]).total_seconds()
            for t in timestamps
        ]
    )

    cmap = LinearSegmentedColormap.from_list(
        "fluke",
        [
            (0.00, "#050000"),
            (0.08, "#180000"),
            (0.18, "#350000"),
            (0.35, "#600000"),
            (0.55, "#a00000"),
            (0.72, "#ff0000"),
            (0.87, "#ff7f00"),
            (0.96, "#ffff00"),
            (1.00, "#ffffff"),
        ],
    )

    Tmin = np.nanmin(frames)
    Tmax = np.nanmax(frames)

    tempdir = Path(tempfile.mkdtemp())

    pngs = []

    for i, frame in enumerate(frames):

        fig, ax = plt.subplots(figsize=(10,7))

        im = ax.imshow(
            frame,
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
            vmin=Tmin,
            vmax=Tmax,
        )

        ax.set_xlabel("Column (pixel)")
        ax.set_ylabel("Row (pixel)")

        ax.set_xticks(np.arange(0, frame.shape[1], tick_step))
        ax.set_yticks(np.arange(0, frame.shape[0], tick_step))

        ax.grid(
            color="white",
            alpha=0.20,
            linewidth=0.5,
        )

        ax.set_title(
            f"{csv_files[i].name}\nElapsed time = {times[i]:.2f} s"
        )

        stats = (
            f"Max : {np.nanmax(frame):5.1f} °C\n"
            f"Avg : {np.nanmean(frame):5.1f} °C\n"
            f"Min : {np.nanmin(frame):5.1f} °C"
        )

        ax.text(
            0.01,
            0.99,
            stats,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            va="top",
            bbox=dict(
                facecolor="black",
                alpha=0.6,
                edgecolor="white",
            ),
        )

        if points is not None:

            for k, (x, y) in enumerate(points):

                if (
                    0 <= x < frame.shape[1]
                    and
                    0 <= y < frame.shape[0]
                ):

                    T = frame[y, x]

                    ax.plot(
                        x,
                        y,
                        marker="+",
                        color="cyan",
                        markersize=16,
                        markeredgewidth=2,
                    )

                    ax.text(
                        x + 1,
                        y - 1,
                        f"P{k+1}\n{T:.1f}°C",
                        color="cyan",
                        fontsize=9,
                        weight="bold",
                    )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature (°C)")

        png = tempdir / f"frame_{i:05d}.png"

        plt.tight_layout()
        plt.savefig(
            png,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        pngs.append(imageio.imread(png))

    imageio.mimsave(
        outfile,
        pngs,
        fps=fps,
        loop=0,
    )

    print(f"Loaded {len(frames)} frames.")
    print(f"Saved GIF to {outfile}")


folder = Path(r"/mnt/c/Documents/thermal_simulation_2/inputs/empiric/Camara termica/Thermal Data 20260717/CICIMAUCR0105-kalinini/images/79-116")

timestamp_pattern = re.compile(
    r"IR_\d+_(\d+-\d+-\d+_\d+\.\d+\.\d+)",
    re.IGNORECASE,
)

def parse_timestamp(filename):
    m = timestamp_pattern.search(filename)
    return datetime.strptime(m.group(1), "%m-%d-%y_%H.%M.%S")

points = [
    (56,46),
    (60,46),
    (65,46),
]

files = sorted(folder.glob("*.csv"))

frames = []
timestamps = []

for f in files:

    df = pd.read_csv(
        f,
        encoding="utf-16",
        skiprows=3,
        header=0,
        index_col=0,
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    frames.append(df.to_numpy(dtype=float))
    timestamps.append(parse_timestamp(f.name))

times = np.array(
    [(t - timestamps[0]).total_seconds() for t in timestamps]
)

save_thermal_gif(
    folder,
    points=points,
    outfile="thermal.gif",
    fps=5,
)
