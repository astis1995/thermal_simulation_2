from pathlib import Path

import pandas as pd

from fitting import fit_temperature_curves
from plotting import plot_temperature_curves


df = pd.read_csv("temperatures.csv")

parameters, fitted = fit_temperature_curves(
    df,
    model="heating",
)

print("\nFit parameters")
print(parameters)

parameters.to_csv(
    "fit_parameters.csv",
    index=False,
)

fitted.to_csv(
    "fit_curves.csv",
    index=False,
)

plot_temperature_curves(
    measured=df,
    fitted=fitted,
    output_folder="plots",
)

print("\nGenerated:")
print("  fit_parameters.csv")
print("  fit_curves.csv")
print("  plots/")
