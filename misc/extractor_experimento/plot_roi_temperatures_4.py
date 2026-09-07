import csv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROUTES_CSV = Path(r"rutas.csv")


# ============================================================
# LOAD EXPERIMENT PATH
# ============================================================

def load_experiment_folder():
    """
    Read experiment_folder from rutas.csv.

    Expected format:

        experiment_folder
        /mnt/c/Documents/thermal_simulation_2/inputs/experimento2
        roi_image_name
        IR_00395.jpg
    """

    if not ROUTES_CSV.exists():
        raise FileNotFoundError(
            f"Routes CSV does not exist:\n{ROUTES_CSV}"
        )

    with open(
        ROUTES_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = [
            row[0].strip()
            for row in csv.reader(f)
            if row and row[0].strip()
        ]

    for i, value in enumerate(rows):

        if value == "experiment_folder":

            if i + 1 >= len(rows):
                raise ValueError(
                    "experiment_folder has no value in rutas.csv."
                )

            return Path(rows[i + 1])

    raise ValueError(
        "rutas.csv does not contain 'experiment_folder'."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load experiment folder
    # --------------------------------------------------------

    experiment_folder = load_experiment_folder()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_csv = (
        experiment_folder
        / "output"
        / "thermal_time_series.csv"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_folder = (
        experiment_folder
        / "output"
        / "plots"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_png = (
        output_folder
        / "roi_temperature_plot.png"
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Thermal time series file not found:\n"
            f"{input_csv}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        input_csv,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Validate time column
    # --------------------------------------------------------

    if "time" not in df.columns:
        raise ValueError(
            "thermal_time_series.csv must contain "
            "a 'time' column."
        )

    # --------------------------------------------------------
    # Parse time
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["time"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid time values were found."
        )

    # --------------------------------------------------------
    # Calculate elapsed time in seconds
    # --------------------------------------------------------

    df["elapsed_seconds"] = (
        df["time"]
        - df["time"].iloc[0]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Identify ROI columns
    #
    # Every column except 'time' is treated as a
    # temperature point.
    # --------------------------------------------------------

    roi_columns = [
        column
        for column in df.columns
        if column not in {
            "time",
            "elapsed_seconds"
        }
    ]

    if not roi_columns:
        raise ValueError(
            "No ROI temperature columns were found."
        )

    # --------------------------------------------------------
    # Convert temperatures to numeric
    # --------------------------------------------------------

    for column in roi_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # Plot every ROI
    # --------------------------------------------------------

    for column in roi_columns:

        plt.plot(
            df["elapsed_seconds"],
            df[column],
            label=column,
            linewidth=1.5
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "Elapsed time (s)"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "ROI Temperature vs Time"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    plt.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    plt.legend(
        loc="best"
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("============================================")
    print("ROI TEMPERATURE PLOT")
    print("============================================")
    print()

    print("Experiment folder:")
    print(experiment_folder)
    print()

    print("Input:")
    print(input_csv)
    print()

    print("ROIs plotted:")

    for roi in roi_columns:
        print(f"  - {roi}")

    print()

    print("Number of data points:")
    print(len(df))

    print()

    print("Output:")
    print(output_png)

    print()
    print("============================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
