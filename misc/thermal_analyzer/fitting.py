"""
fitting.py

Curve fitting utilities for thermal experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def heating_model(t, T0, A, k):
    """
    Heating model

    T(t) = T0 + A * (1 - exp(-k t))
    """
    return T0 + A * (1.0 - np.exp(-k * t))


def cooling_model(t, T_inf, T0, k):
    """
    Cooling model

    T(t) = T_inf + (T0 - T_inf) exp(-k t)
    """
    return T_inf + (T0 - T_inf) * np.exp(-k * t)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def coefficient_of_determination(y, y_fit):
    """
    Compute coefficient of determination (R²).
    """

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Fit one curve
# ----------------------------------------------------------------------

def fit_point(
    time,
    temperature,
    model="heating",
):
    """
    Fit a single temperature curve.

    Parameters
    ----------
    time : array-like

    temperature : array-like

    model : {"heating","cooling"}

    Returns
    -------
    parameters : dict

    fitted_curve : ndarray
    """

    time = np.asarray(time, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    mask = np.isfinite(time) & np.isfinite(temperature)

    time = time[mask]
    temperature = temperature[mask]

    if len(time) < 4:
        raise RuntimeError("Not enough data points to fit.")

    if model == "heating":

        p0 = (
            temperature[0],
            temperature.max() - temperature[0],
            0.01,
        )

        bounds = (
            [-np.inf, 0.0, 0.0],
            [np.inf, np.inf, np.inf],
        )

        popt, _ = curve_fit(
            heating_model,
            time,
            temperature,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        y_fit = heating_model(time, *popt)

        result = {
            "T0": popt[0],
            "A": popt[1],
            "k": popt[2],
        }

    elif model == "cooling":

        p0 = (
            temperature[-1],
            temperature[0],
            0.01,
        )

        bounds = (
            [-np.inf, -np.inf, 0.0],
            [np.inf, np.inf, np.inf],
        )

        popt, _ = curve_fit(
            cooling_model,
            time,
            temperature,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        y_fit = cooling_model(time, *popt)

        result = {
            "T_inf": popt[0],
            "T0": popt[1],
            "k": popt[2],
        }

    else:
        raise ValueError(f"Unknown model '{model}'")

    result["R2"] = coefficient_of_determination(
        temperature,
        y_fit,
    )

    return result, y_fit


# ----------------------------------------------------------------------
# Fit complete dataframe
# ----------------------------------------------------------------------

def fit_temperature_curves(
    dataframe,
    model="heating",
):
    """
    Fit every temperature column in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame

        Time | P1 | P2 | ...

    model : {"heating","cooling"}

    Returns
    -------
    parameters_df : DataFrame

    fitted_df : DataFrame
    """

    if "Time" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'Time' column.")

    time = dataframe["Time"].to_numpy()

    fitted_df = pd.DataFrame()
    fitted_df["Time"] = time

    results = []

    for column in dataframe.columns:

        if column == "Time":
            continue

        params, y_fit = fit_point(
            time,
            dataframe[column].to_numpy(),
            model=model,
        )

        params["Point"] = column

        results.append(params)

        fitted_df[column] = y_fit

    parameters_df = pd.DataFrame(results)

    # Put Point first
    cols = ["Point"] + [c for c in parameters_df.columns if c != "Point"]
    parameters_df = parameters_df[cols]

    return parameters_df, fitted_df
