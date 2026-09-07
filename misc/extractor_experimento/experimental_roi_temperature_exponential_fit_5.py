import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


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
# EXPONENTIAL MODEL WITH DEAD TIME
# ============================================================

def exponential_dead_time(t, A, tau, T):
    """
    Exponential heating model with dead time.

        f(t) = 0                              t <= T

        f(t) = A * (1 - exp(-(t-T)/tau))      t > T

    Parameters
    ----------
    A   : asymptotic temperature increase
    tau : time constant [s]
    T   : dead time [s]
    """

    t = np.asarray(t)

    result = np.zeros_like(
        t,
        dtype=float
    )

    active = t > T

    result[active] = (
        A
        * (
            1.0
            - np.exp(
                -(t[active] - T) / tau
            )
        )
    )

    return result


# ============================================================
# FIT ONE ROI
# ============================================================

def fit_roi(t, temperature):
    """
    Fit A, tau and T using nonlinear least squares.
    """

    # --------------------------------------------------------
    # Remove NaN values
    # --------------------------------------------------------

    mask = (
        np.isfinite(t)
        & np.isfinite(temperature)
    )

    t_fit = np.asarray(t)[mask]
    y_fit = np.asarray(temperature)[mask]

    if len(t_fit) < 5:
        raise ValueError(
            "Not enough valid data points."
        )

    # --------------------------------------------------------
    # Estimate initial parameters
    # --------------------------------------------------------

    y_min = np.min(y_fit)
    y_max = np.max(y_fit)

    A_initial = y_max - y_min

    if A_initial <= 0:
        A_initial = max(
            abs(y_max),
            1.0
        )

    # Initial tau: approximately 1/3 of experiment duration
    duration = t_fit[-1] - t_fit[0]

    tau_initial = max(
        duration / 3.0,
        1.0
    )

    # Initial dead time
    T_initial = max(
        0.0,
        duration * 0.05
    )

    # --------------------------------------------------------
    # Shift temperature so the model starts at zero
    # --------------------------------------------------------

    baseline = np.median(
        y_fit[:max(3, len(y_fit) // 20)]
    )

    y_relative = y_fit - baseline

    # --------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------

    A_upper = max(
        np.max(y_relative) * 5.0,
        A_initial * 5.0,
        1.0
    )

    # T cannot be later than the last measurement
    T_upper = max(
        t_fit[-1] * 0.95,
        0.1
    )

    tau_upper = max(
        duration * 10.0,
        10.0
    )

    lower_bounds = [
        0.0,       # A
        0.01,      # tau
        0.0        # T
    ]

    upper_bounds = [
        A_upper,
        tau_upper,
        T_upper
    ]

    # --------------------------------------------------------
    # Residual function
    # --------------------------------------------------------

    def residuals(params):

        A, tau, T = params

        prediction = exponential_dead_time(
            t_fit,
            A,
            tau,
            T
        )

        return prediction - y_relative

    # --------------------------------------------------------
    # Perform fit
    # --------------------------------------------------------

    result = least_squares(
        residuals,
        x0=[
            A_initial,
            tau_initial,
            T_initial
        ],
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        max_nfev=10000
    )

    A, tau, T = result.x

    # --------------------------------------------------------
    # Calculate R²
    # --------------------------------------------------------

    prediction = exponential_dead_time(
        t_fit,
        A,
        tau,
        T
    )

    ss_res = np.sum(
        (y_relative - prediction) ** 2
    )

    ss_tot = np.sum(
        (y_relative - np.mean(y_relative)) ** 2
    )

    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = np.nan

    return {
        "A": A,
        "tau": tau,
        "T": T,
        "baseline": baseline,
        "R2": r_squared
    }


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
        / "roi_temperature_exponential_fit.png"
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
    # Read data
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
    # Parse timestamps
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["time"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid time values were found."
        )

    # --------------------------------------------------------
    # Elapsed time
    # --------------------------------------------------------

    df["elapsed_seconds"] = (
        df["time"]
        - df["time"].iloc[0]
    ).dt.total_seconds()

    t = df["elapsed_seconds"].values

    # --------------------------------------------------------
    # Identify ROI columns
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

    # ========================================================
    # CREATE PLOT
    # ========================================================

    plt.figure(
        figsize=(13, 8)
    )

    fit_results = {}

    # --------------------------------------------------------
    # Fit and plot every ROI
    # --------------------------------------------------------

    for roi in roi_columns:

        df[roi] = pd.to_numeric(
            df[roi],
            errors="coerce"
        )

        temperature = df[roi].values

        try:

            result = fit_roi(
                t,
                temperature
            )

            fit_results[roi] = result

            # ------------------------------------------------
            # Experimental data
            # ------------------------------------------------

            plt.plot(
                t,
                temperature,
                linewidth=1.2,
                label=roi
            )

            # ------------------------------------------------
            # Fitted curve
            # ------------------------------------------------

            t_smooth = np.linspace(
                t.min(),
                t.max(),
                1000
            )

            fitted_temperature = (
                result["baseline"]
                + exponential_dead_time(
                    t_smooth,
                    result["A"],
                    result["tau"],
                    result["T"]
                )
            )

            plt.plot(
                t_smooth,
                fitted_temperature,
                linestyle="--",
                linewidth=2.0
            )

            print()
            print(
                f"{roi}:"
            )

            print(
                f"    A   = "
                f"{result['A']:.3f} °C"
            )

            print(
                f"    tau = "
                f"{result['tau']:.3f} s"
            )

            print(
                f"    T   = "
                f"{result['T']:.3f} s"
            )

            print(
                f"    R²  = "
                f"{result['R2']:.5f}"
            )

        except Exception as e:

            print()
            print(
                f"WARNING: Could not fit {roi}:"
            )

            print(
                f"    {e}"
            )

            # Plot experimental data anyway

            plt.plot(
                t,
                temperature,
                linewidth=1.2,
                label=roi
            )

    # ========================================================
    # LEGEND WITH FIT PARAMETERS
    # ========================================================

    handles, labels = plt.gca().get_legend_handles_labels()

    new_labels = []

    for label in labels:

        if label in fit_results:

            result = fit_results[label]

            new_labels.append(
                f"{label}: "
                f"τ={result['tau']:.1f} s, "
                f"T={result['T']:.1f} s"
            )

        else:

            new_labels.append(label)

    plt.legend(
        handles,
        new_labels,
        loc="best"
    )

    # ========================================================
    # AXES
    # ========================================================

    plt.xlabel(
        "Elapsed time (s)"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "ROI Temperature with Exponential Heating Fit"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        "EXPONENTIAL FIT COMPLETE"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Input:"
    )

    print(
        input_csv
    )

    print()

    print(
        "Fitted ROIs:"
    )

    for roi, result in fit_results.items():

        print(
            f"  {roi}: "
            f"A={result['A']:.3f} °C, "
            f"tau={result['tau']:.3f} s, "
            f"T={result['T']:.3f} s, "
            f"R²={result['R2']:.5f}"
        )

    print()

    print(
        "Output:"
    )

    print(
        output_png
    )

    print()

    print(
        "============================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
