#!/usr/bin/env python3

import itertools
import subprocess
import sys
from pathlib import Path
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

# Directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Base parameter file
BASE_CONFIG = BASE_DIR / "parameters_viga_aluminio_04.yaml"

# Folder where the 81 generated YAML files will be stored
CONFIG_DIR = BASE_DIR / "parameter_sweep_configs"

# Orchestrator
ORCHESTRATOR = BASE_DIR / "orchestrator.py"

# Simulation name passed to orchestrator
SIMULATION_NAME = "simulation2"

# Extractor
EXTRACTOR = (
    BASE_DIR
    / "misc"
    / "extractor_experimento"
    / "simulation_roi_temperature_exponential_fit_6.py"
)

# ------------------------------------------------------------
# Parameters to sweep
# ------------------------------------------------------------

BASE_VALUES = {
    "h": 5.0,
    "k": 237.0,
    "c": 900.0,
    "rho": 2699.0,
}

FACTORS = [0.1, 1.0, 10.0]


# ============================================================
# FUNCTIONS
# ============================================================

def modify_config(config, h, k, c, rho):
    """
    Modify the physical parameters in a YAML configuration.
    """

    config["domain"]["material"]["k"] = k
    config["domain"]["material"]["rho"] = rho
    config["domain"]["material"]["c"] = c

    config["simulation"]["physics"]["convection"]["h"] = h

    return config


def format_factor(factor):
    """
    Convert numerical factor to a filename-friendly string.
    """

    if factor == 0.1:
        return "0p1"

    if factor == 1.0:
        return "1"

    if factor == 10.0:
        return "10"

    return str(factor).replace(".", "p")


def generate_configs():
    """
    Generate all combinations of h, k, c and rho.
    """

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG, "r") as f:
        base_config = yaml.safe_load(f)

    combinations = list(
        itertools.product(
            FACTORS,  # h
            FACTORS,  # k
            FACTORS,  # c
            FACTORS,  # rho
        )
    )

    print("=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)

    print(f"Base configuration : {BASE_CONFIG}")
    print(f"Output directory   : {CONFIG_DIR}")
    print(f"Number of runs     : {len(combinations)}")
    print()

    generated_files = []

    for index, (fh, fk, fc, frho) in enumerate(combinations, start=1):

        h = BASE_VALUES["h"] * fh
        k = BASE_VALUES["k"] * fk
        c = BASE_VALUES["c"] * fc
        rho = BASE_VALUES["rho"] * frho

        config = modify_config(
            # Deep copy so combinations never modify each other
            __import__("copy").deepcopy(base_config),
            h,
            k,
            c,
            rho,
        )

        filename = (
            f"parameters_viga_aluminio_04_"
            f"h{format_factor(fh)}_"
            f"k{format_factor(fk)}_"
            f"c{format_factor(fc)}_"
            f"rho{format_factor(frho)}.yaml"
        )

        output_file = CONFIG_DIR / filename

        # Add useful comments/documentation to the YAML itself
        config["simulation"]["physics"]["convection"]["h"] = h

        with open(output_file, "w") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
                default_flow_style=False,
            )

        generated_files.append(output_file)

        print(
            f"[{index:02d}/81] "
            f"h={h:g}, "
            f"k={k:g}, "
            f"c={c:g}, "
            f"rho={rho:g}"
        )

    print()
    print(f"Generated {len(generated_files)} configuration files.")

    return generated_files


def run_simulation(config_file, index, total):
    """
    Run one FEniCSx simulation.
    """

    print()
    print("=" * 70)
    print(f"SIMULATION {index}/{total}")
    print("=" * 70)
    print(f"Configuration: {config_file.name}")
    print()

    command = [
        sys.executable,
        str(ORCHESTRATOR),
        SIMULATION_NAME,
        str(config_file),
    ]

    print("Running:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        print()
        print("=" * 70)
        print("SIMULATION FAILED")
        print("=" * 70)
        print(f"Configuration: {config_file}")
        print(f"Return code: {result.returncode}")

        sys.exit(result.returncode)


def run_extractor():
    """
    Run the ROI temperature exponential fitting script
    after all simulations have completed.
    """

    print()
    print("=" * 70)
    print("RUNNING ROI EXTRACTION / EXPONENTIAL FIT")
    print("=" * 70)
    print()

    extractor_dir = EXTRACTOR.parent

    command = [
        sys.executable,
        str(EXTRACTOR),
    ]

    print("Running:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=extractor_dir,
    )

    if result.returncode != 0:
        print()
        print("=" * 70)
        print("EXTRACTOR FAILED")
        print("=" * 70)
        print(f"Return code: {result.returncode}")

        sys.exit(result.returncode)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        BASE_CONFIG,
        ORCHESTRATOR,
        EXTRACTOR,
    ]

    for file in required_files:
        if not file.exists():
            print(f"ERROR: File not found:")
            print(f"  {file}")
            sys.exit(1)

    # --------------------------------------------------------
    # Generate YAML files
    # --------------------------------------------------------

    configs = generate_configs()

    # --------------------------------------------------------
    # Run simulations
    # --------------------------------------------------------

    total = len(configs)

    print()
    print("=" * 70)
    print(f"STARTING {total} SIMULATIONS")
    print("=" * 70)

    for index, config_file in enumerate(configs, start=1):
        run_simulation(
            config_file,
            index,
            total,
        )

    # --------------------------------------------------------
    # All simulations completed
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL SIMULATIONS COMPLETED")
    print("=" * 70)

    # --------------------------------------------------------
    # Run extractor
    # --------------------------------------------------------

    run_extractor()

    print()
    print("=" * 70)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
