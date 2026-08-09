"""
plotting.py

Visualization utilities for thermal experiments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_temperature_curves(
    measured,
    fitted,
    output_folder="plots",
):
    """
    Plot measured temperatures together with fitted curves.

    Parameters
    ----------
    measured : DataFrame

    fitted : DataFrame

    output_folder : str or Path
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    time = measured["Time"]

    for column in measured.columns:

        if column == "Time":
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            time,
            measured[column],
            "o",
            label="Measured",
        )

        ax.plot(
            time,
            fitted[column],
            "-",
            linewidth=2,
            label="Fit",
        )

        ax.set_title(column)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (°C)")

        ax.grid(True)

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_folder / f"{column}.png",
            dpi=300,
        )

        plt.close(fig)
