"""
temperature_extractor.py

Extract temperatures from all IR CSV files.
"""

import numpy as np
import pandas as pd

from io_utils import (
    load_temperature_csv,
    get_timestamp,
)

from datetime import datetime


def extract_temperature(matrix, row, col, radius=0):
    """
    Extract the temperature around a point.

    Parameters
    ----------
    matrix : np.ndarray

    row : int

    col : int

    radius : int
        0 -> 1x1
        1 -> 3x3
        2 -> 5x5
        ...

    Returns
    -------
    float
    """

    rows, cols = matrix.shape

    r0 = max(0, row - radius)
    r1 = min(rows, row + radius + 1)

    c0 = max(0, col - radius)
    c1 = min(cols, col + radius + 1)

    window = matrix[r0:r1, c0:c1]

    return float(np.mean(window))


def load_temperature_series(
    files,
    points,
    averaging_radius,
    timestamp_priority=None,
):
    """
    Extract all temperatures for all selected points.

    Parameters
    ----------
    files : list[IRFile]

    points : list

    averaging_radius : int

    timestamp_priority : list

    Returns
    -------
    pandas.DataFrame

    Columns

        Time
        P1
        P2
        ...
    """

    rows = []

    first_timestamp = None

    total = len(files)

    for i, ir in enumerate(files):

        print(f"\rProcessing {i+1}/{total}: {ir.stem}", end="")

        matrix, metadata = load_temperature_csv(ir.csv)

        # -------------------------------------------------------
        # Temporary timestamp handling
        # -------------------------------------------------------

        timestamp = get_timestamp(
            ir_file=ir,
            metadata=metadata,
            priority=timestamp_priority,
        )

        if first_timestamp is None:
            first_timestamp = timestamp

        elapsed = (timestamp - first_timestamp).total_seconds()

        record = {
            "Time": elapsed
        }

        for point in points:

            value = extract_temperature(
                matrix,
                point["row"],
                point["col"],
                averaging_radius,
            )

            record[point["name"]] = value

        rows.append(record)

    print()

    df = pd.DataFrame(rows)

    df = df.sort_values("Time")

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df
