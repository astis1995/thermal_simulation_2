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
# LOAD ROUTES
# ============================================================

def load_routes():
    """
    Read simulation information from rutas.csv.

    Expected format:

        experiment_folder
        /mnt/c/Documents/thermal_simulation_2/inputs/experimento2

        roi_image_name
        IR_00365.jpg

        simulation_name
        simulation2

        simulation_folder
        /mnt/c/Documents/thermal_simulation_2/outputs/simulation2
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

    routes = {}

    for i, value in enumerate(rows):

        if value in {
            "experiment_folder",
            "roi_image_name",
            "simulation_name",
            "simulation_folder"
        }:

            if i + 1 >= len(rows):
                raise ValueError(
                    f"{value} has no value in rutas.csv."
                )

            routes[value] = rows[i + 1]

    required = {
        "experiment_folder",
        "roi_image_name",
        "simulation_name",
        "simulation_folder"
    }

    missing = required - set(routes)

    if missing:
        raise ValueError(
            "Missing entries in rutas.csv:\n"
            + "\n".join(sorted(missing))
        )

    return routes


# ============================================================
# FIND ALL ROI CSV FILES
# ============================================================
def find_simulation_roi_files(
    simulation_folder,
    simulation_name
):
    """
    Search:

        simulation_folder/
            simulation_name/
                results/

    for files ending with '-roi.csv'.

    Return all ROI files sorted by filename.
    """

    results_folder = (
        Path(simulation_folder)
        / simulation_name
        / "results"
    )

    if not results_folder.exists():
        raise FileNotFoundError(
            f"Simulation results folder does not exist:\n"
            f"{results_folder}"
        )

    roi_files = sorted(
        file
        for file in results_folder.iterdir()
        if (
            file.is_file()
            and file.name.endswith("-roi.csv")
        )
    )

    if not roi_files:
        raise FileNotFoundError(
            "No file ending in '-roi.csv' was found in:\n"
            f"{results_folder}"
        )

    return roi_files

# ============================================================
# EXPONENTIAL MODEL WITH DEAD TIME
# ============================================================

def exponential_dead_time(
    t,
    A,
    tau,
    T
):

    """
    Model:

        f(t) = 0                              t <= T

        f(t) = A * (1-exp(-(t-T)/tau))        t > T

    A   = temperature increase
    tau = time constant [s]
    T   = dead time [s]
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
# FIT ONE SIMULATION POINT
# ============================================================

def fit_series(
    t,
    temperature
):

    """
    Fit:

        Temperature =
            baseline +
            A * (1-exp(-(t-T)/tau))
    """

    t = np.asarray(
        t,
        dtype=float
    )

    temperature = np.asarray(
        temperature,
        dtype=float
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    mask = (
        np.isfinite(t)
        & np.isfinite(temperature)
    )

    t = t[mask]
    temperature = temperature[mask]

    if len(t) < 5:
        raise ValueError(
            "Not enough valid data points."
        )

    # --------------------------------------------------------
    # Sort by time
    # --------------------------------------------------------

    order = np.argsort(t)

    t = t[order]
    temperature = temperature[order]

    # --------------------------------------------------------
    # Start time at zero
    # --------------------------------------------------------

    t = t - t[0]

    duration = t[-1]

    if duration <= 0:
        raise ValueError(
            "Time range is zero."
        )

    # --------------------------------------------------------
    # Initial baseline
    # --------------------------------------------------------

    n_initial = max(
        3,
        min(
            20,
            len(temperature) // 20
        )
    )

    baseline = np.median(
        temperature[:n_initial]
    )

    y = temperature - baseline

    # --------------------------------------------------------
    # Initial A
    # --------------------------------------------------------

    A_initial = (
        np.max(y)
        - np.min(y)
    )

    if A_initial <= 0:
        A_initial = 1.0

    # --------------------------------------------------------
    # Initial tau
    # --------------------------------------------------------

    tau_initial = max(
        duration / 3.0,
        0.1
    )

    # --------------------------------------------------------
    # Initial dead time
    # --------------------------------------------------------

    T_initial = max(
        0.0,
        duration * 0.05
    )

    # --------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------

    A_upper = max(
        A_initial * 10.0,
        np.max(y) * 10.0,
        1.0
    )

    tau_upper = max(
        duration * 20.0,
        1.0
    )

    T_upper = duration * 0.95

    lower_bounds = [
        0.0,       # A
        0.001,     # tau
        0.0        # T
    ]

    upper_bounds = [
        A_upper,
        tau_upper,
        T_upper
    ]

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    def residuals(params):

        A, tau, T = params

        prediction = exponential_dead_time(
            t,
            A,
            tau,
            T
        )

        return prediction - y

    # --------------------------------------------------------
    # Fit
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
        max_nfev=20000
    )

    A, tau, T = result.x

    # --------------------------------------------------------
    # Calculate R²
    # --------------------------------------------------------

    prediction = exponential_dead_time(
        t,
        A,
        tau,
        T
    )

    ss_res = np.sum(
        (y - prediction) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:

        r_squared = (
            1.0
            - ss_res / ss_tot
        )

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

    # ========================================================
    # LOAD ROUTES
    # ========================================================

    routes = load_routes()

    simulation_name = (
        routes["simulation_name"]
    )

    simulation_folder = Path(
        routes["simulation_folder"]
    )

    # ========================================================
    # FIND ALL SIMULATION ROI FILES
    # ========================================================

    simulation_csv_files = find_simulation_roi_files(
        simulation_folder,
        simulation_name
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    experiment_folder = Path(
        routes["experiment_folder"]
    )

    output_folder = (
        experiment_folder
        / "output"
        / "plots"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PROCESS EACH ROI CSV FILE
    # ========================================================

    print()
    print("============================================")
    print("SIMULATION ROI EXPONENTIAL FIT")
    print("============================================")
    print()

    print(f"Found {len(simulation_csv_files)} ROI files.")

    for simulation_csv in simulation_csv_files:

        print()
        print("--------------------------------------------")
        print(f"Processing: {simulation_csv.name}")
        print("--------------------------------------------")

        # Output image uses EXACTLY the same timestamp/name
        # as the input ROI CSV, with .png instead of .csv.
        output_png = (
            output_folder
            / f"{simulation_csv.stem}.png"
        )

        # ====================================================
        # READ SIMULATION CSV
        # ====================================================

        df = pd.read_csv(
            simulation_csv,
            encoding="utf-8-sig"
        )

        # ====================================================
        # TIME
        # ====================================================

        if "time_s" in df.columns:

            time = pd.to_numeric(
                df["time_s"],
                errors="coerce"
            ).values

        elif "time" in df.columns:

            time = pd.to_numeric(
                df["time"],
                errors="coerce"
            ).values

        else:

            print(
                "WARNING: Simulation CSV does not contain "
                "'time_s' or 'time'. Skipping."
            )
            continue

        # ====================================================
        # FIND TEMPERATURE POINTS
        # ====================================================

        roi_columns = [
            column
            for column in df.columns
            if column.endswith(".temperature")
        ]

        if not roi_columns:
            print(
                "WARNING: No '.temperature' columns were found "
                "in this simulation CSV. Skipping."
            )
            continue

        print("Points:")

        for roi in roi_columns:
            print(f"  - {roi}")

        # ====================================================
        # CREATE FIGURE
        # ====================================================

        plt.figure(
            figsize=(14, 9)
        )

        fit_results = {}

        # ====================================================
        # PROCESS EACH POINT
        # ====================================================

        for roi in roi_columns:

            temperature = pd.to_numeric(
                df[roi],
                errors="coerce"
            ).values

            # ------------------------------------------------
            # Fit
            # ------------------------------------------------

            try:

                result = fit_series(
                    time,
                    temperature
                )

                fit_results[roi] = result

            except Exception as e:

                print()
                print(
                    f"WARNING: Could not fit {roi}:"
                )

                print(
                    f"    {e}"
                )

                plt.plot(
                    time,
                    temperature,
                    linewidth=1.5,
                    label=roi
                )

                continue

            # ------------------------------------------------
            # Simulation data
            # ------------------------------------------------

            plt.plot(
                time,
                temperature,
                linewidth=1.3,
                label=roi
            )

            # ------------------------------------------------
            # Smooth fitted curve
            # ------------------------------------------------

            valid_time = time[np.isfinite(time)]

            if len(valid_time) == 0:
                continue

            t_smooth = np.linspace(
                np.nanmin(time),
                np.nanmax(time),
                1000
            )

            fitted_temperature = (
                result["baseline"]
                + exponential_dead_time(
                    t_smooth - t_smooth[0],
                    result["A"],
                    result["tau"],
                    result["T"]
                )
            )

            plt.plot(
                t_smooth,
                fitted_temperature,
                linestyle="--",
                linewidth=2.0,
                label=(
                    f"{roi} fit "
                    f"(τ={result['tau']:.1f} s, "
                    f"T={result['T']:.1f} s)"
                )
            )

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print()
            print(
                f"{roi}"
            )

            print(
                f"    A   = {result['A']:.4f} °C"
            )

            print(
                f"    τ   = {result['tau']:.4f} s"
            )

            print(
                f"    T   = {result['T']:.4f} s"
            )

            print(
                f"    R²  = {result['R2']:.6f}"
            )

        # ====================================================
        # AXES
        # ====================================================

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Temperature (K)"
        )

        plt.title(
            f"Simulation ROI Temperatures with Exponential Fits\n"
            f"{simulation_csv.name}"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        # ====================================================
        # LEGEND
        # ====================================================

        plt.legend(
            loc="best",
            fontsize=9
        )

        plt.tight_layout()

        # ====================================================
        # SAVE
        # ====================================================

        plt.savefig(
            output_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print()
        print("Output:")
        print(output_png)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("============================================")
    print("SIMULATION ROI EXPONENTIAL FIT")
    print("============================================")
    print()

    print("Simulation:")
    print(simulation_csv)

    print()

    print("Points:")

    for roi in roi_columns:
        print(f"  - {roi}")

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    plt.figure(
        figsize=(14, 9)
    )

    fit_results = {}

    # ========================================================
    # PROCESS EACH POINT
    # ========================================================

    for roi in roi_columns:

        temperature = pd.to_numeric(
            df[roi],
            errors="coerce"
        ).values

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        try:

            result = fit_series(
                time,
                temperature
            )

            fit_results[roi] = result

        except Exception as e:

            print()
            print(
                f"WARNING: Could not fit {roi}:"
            )

            print(
                f"    {e}"
            )

            plt.plot(
                time,
                temperature,
                linewidth=1.5,
                label=roi
            )

            continue

        # ----------------------------------------------------
        # Experimental simulation data
        # ----------------------------------------------------

        plt.plot(
            time,
            temperature,
            linewidth=1.3,
            label=roi
        )

        # ----------------------------------------------------
        # Smooth fitted curve
        # ----------------------------------------------------

        t_smooth = np.linspace(
            np.nanmin(time),
            np.nanmax(time),
            1000
        )

        fitted_temperature = (
            result["baseline"]
            + exponential_dead_time(
                t_smooth - t_smooth[0],
                result["A"],
                result["tau"],
                result["T"]
            )
        )

        plt.plot(
            t_smooth,
            fitted_temperature,
            linestyle="--",
            linewidth=2.0,
            label=(
                f"{roi} fit "
                f"(τ={result['tau']:.1f} s, "
                f"T={result['T']:.1f} s)"
            )
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()
        print(
            f"{roi}"
        )

        print(
            f"    A   = {result['A']:.4f} °C"
        )

        print(
            f"    τ   = {result['tau']:.4f} s"
        )

        print(
            f"    T   = {result['T']:.4f} s"
        )

        print(
            f"    R²  = {result['R2']:.6f}"
        )

    # ========================================================
    # AXES
    # ========================================================

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "Simulation ROI Temperatures "
        "with Exponential Fits"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    # ========================================================
    # LEGEND
    # ========================================================

    plt.legend(
        loc="best",
        fontsize=9
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
    # SUMMARY
    # ========================================================

    print()
    print("============================================")
    print("FIT COMPLETE")
    print("============================================")
    print()
    print(f"Processed {len(simulation_csv_files)} ROI files.")
    print(f"Output folder:")
    print(output_folder)
    print()
    print("============================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
