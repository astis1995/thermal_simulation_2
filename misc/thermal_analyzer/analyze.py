from pathlib import Path

from config import load_config

from io_utils import find_ir_files
from point_selector import load_or_select_points
from temperature_extractor import load_temperature_series
from fitting import fit_temperature_curves
from plotting import plot_temperature_curves


def main():

    # -------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------

    cfg = load_config("experiment1.yaml")

    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Locate experiment files
    # -------------------------------------------------------

    files = find_ir_files(input_folder)

    if not files:
        raise RuntimeError(f"No IR files found in {input_folder}")

    # -------------------------------------------------------
    # Select (or load) measurement points
    # -------------------------------------------------------

    points = load_or_select_points(
        files=files,
        point_file=output_folder / cfg.points.file,
        force_reselect=cfg.points.force_reselect,
    )

    # -------------------------------------------------------
    # Extract temperatures
    # -------------------------------------------------------

    temperatures = load_temperature_series(
        files=files,
        points=points,
        averaging_radius=cfg.averaging_radius,
        timestamp_priority=cfg.timestamp_priority,
    )

    temperatures_file = output_folder / "temperatures.csv"

    temperatures.to_csv(
        temperatures_file,
        index=False,
    )

    # -------------------------------------------------------
    # Fit curves
    # -------------------------------------------------------

    parameters, fitted = fit_temperature_curves(
        temperatures,
        model=cfg.fit_model,
    )

    parameters.to_csv(
        output_folder / "fit_parameters.csv",
        index=False,
    )

    fitted.to_csv(
        output_folder / "fit_curves.csv",
        index=False,
    )

    # -------------------------------------------------------
    # Create plots
    # -------------------------------------------------------

    plot_temperature_curves(
        measured=temperatures,
        fitted=fitted,
        output_folder=output_folder / "plots",
    )

    print("\nAnalysis complete.")
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
