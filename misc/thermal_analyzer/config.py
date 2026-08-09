"""
config.py

Configuration loader.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


DEFAULT_CONFIG = Path("experiment1.yaml")


def _to_namespace(obj):
    """
    Recursively convert dictionaries into objects with attribute access.
    """

    if isinstance(obj, dict):
        return SimpleNamespace(
            **{k: _to_namespace(v) for k, v in obj.items()}
        )

    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]

    return obj


def load_config(config_file=DEFAULT_CONFIG):
    """
    Load and validate the YAML configuration.
    """

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_file}"
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    return _to_namespace(config)


def validate_config(config):
    """
    Validate the experiment configuration.
    """

    required = [
        "input_folder",
        "output_folder",
        "fit_model",
        "temperature_unit",
        "timestamp_priority",
        "points",
    ]

    for key in required:
        if key not in config:
            raise ValueError(
                f"Missing configuration entry '{key}'"
            )

    point_required = [
        "mode",
        "file",
        "force_reselect",
    ]

    for key in point_required:
        if key not in config["points"]:
            raise ValueError(
                f"Missing points configuration '{key}'"
            )

    return True
