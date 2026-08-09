from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ----------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------

results_folder = Path(
    r"/mnt/c/Documents/thermal_simulation_2/outputs/simulation1/results"
)

csv_file = results_folder / "simulation1-20260720-222309-roi.csv"

t_start = 5.0      # Ignore everything before this time (s)

# ----------------------------------------------------

df = pd.read_csv(csv_file)

time = df["time"].to_numpy()

temperature_columns = [
    c for c in df.columns
    if c != "time"
]


def heating_model(t, Tmin, Trise, tau):
    return Tmin + Trise * (1 - np.exp(-t / tau))


for column in temperature_columns:

    temperatures = df[column].to_numpy()

    mask = np.isfinite(temperatures) & (time >= t_start)

    t = time[mask]
    T = temperatures[mask]

    # Shift the time axis so heating starts at t=0
    t = t - t_start

    Tmin_guess = T.min()
    Trise_guess = T.max() - T.min()
    tau_guess = max(t[-1] / 3, 1)

    try:

        popt, _ = curve_fit(
            heating_model,
            t,
            T,
            p0=[Tmin_guess, Trise_guess, tau_guess],
            bounds=(
                [-1000, 0, 1e-6],
                [5000, 5000, np.inf],
            ),
        )

        Tmin, Trise, tau = popt

        fit = heating_model(t, *popt)

        print(
            f"{column:25s}"
            f"Tmin={Tmin:.3f}  "
            f"Trise={Trise:.3f}  "
            f"tau={tau:.3f} s"
        )

        plt.figure(figsize=(7,5))

        plt.plot(
            t,
            T,
            "o",
            markersize=3,
            label="Simulation",
        )

        plt.plot(
            t,
            fit,
            linewidth=2,
            label=f"Fit (τ={tau:.3f} s)",
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (K)")
        plt.title(column)

        plt.grid(True)
        plt.legend()

    except RuntimeError:

        print(f"Fit failed for {column}")

plt.show()
