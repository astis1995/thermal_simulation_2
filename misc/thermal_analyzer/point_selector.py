"""
point_selector.py

Interactive point selection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import json

from io_utils import load_image, load_temperature_csv

def select_points(image, csv_shape, max_points=10):
    """
    Display an image and allow the user to select points.

    Parameters
    ----------
    image : PIL.Image.Image
        Thermal JPG image.

    csv_shape : tuple
        Shape of the temperature matrix (rows, cols).

    max_points : int
        Maximum number of selectable points.

    Returns
    -------
    list of dict

    Example
    -------
    [
        {
            "name": "P1",
            "row": 35,
            "col": 82
        },
        ...
    ]
    """

    rows, cols = csv_shape
    width, height = image.size

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)
    ax.set_title(
        f"Select up to {max_points} points\n"
        "Press ENTER when finished."
    )
    ax.set_axis_off()

    clicks = plt.ginput(
        n=max_points,
        timeout=0
    )

    plt.close(fig)

    points = []

    for i, (x, y) in enumerate(clicks):

        col = int(np.clip(round(x * cols / width), 0, cols - 1))
        row = int(np.clip(round(y * rows / height), 0, rows - 1))

        points.append(
            {
                "name": f"P{i+1}",
                "row": row,
                "col": col,
            }
        )

    return points


def save_selected_points_image(
    image,
    points,
    csv_shape,
    output_file,
):
    """
    Save an image showing the selected points.

    Parameters
    ----------
    image : PIL.Image.Image

    points : list

    csv_shape : tuple
        (rows, cols)

    output_file : str or Path
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = csv_shape
    width, height = image.size

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)

    for point in points:

        x = point["col"] * width / cols
        y = point["row"] * height / rows

        ax.plot(
            x,
            y,
            "ro",
            markersize=8,
        )

        ax.text(
            x + 6,
            y + 6,
            point["name"],
            color="white",
            fontsize=10,
            bbox=dict(
                facecolor="red",
                edgecolor="none",
                alpha=0.7,
            ),
        )

    ax.set_axis_off()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def load_or_select_points(
    files,
    point_file,
    force_reselect=False,
):
    """
    Load previously selected points if they exist.
    Otherwise ask the user to select them and save them.
    """

    point_file = Path(point_file)

    if point_file.exists() and not force_reselect:

        with open(point_file, "r") as f:
            return json.load(f)

    if len(files) == 0:
        raise RuntimeError("No experiment files found.")

    first = files[0]

    image = load_image(first.jpg)
    temperature, metadata = load_temperature_csv(first.csv)

    points = select_points(
        image=image,
        csv_shape=temperature.shape,
    )

    point_file.parent.mkdir(parents=True, exist_ok=True)

    with open(point_file, "w") as f:
        json.dump(points, f, indent=4)

    save_selected_points_image(
        image=image,
        points=points,
        csv_shape=temperature.shape,
        output_file=point_file.with_suffix(".png"),
    )

    return points
