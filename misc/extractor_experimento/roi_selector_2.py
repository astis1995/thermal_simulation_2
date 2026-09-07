import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CSV containing the experiment folder
#
# Example:
#
# experiment_folder
# /mnt/c/Users/esteb/Downloads/output/experimento2
#
# ------------------------------------------------------------

ROUTES_CSV = Path(
    r"rutas.csv"
)


# ------------------------------------------------------------
# Image extensions
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# THERMAL CSV DIMENSIONS
# ============================================================

# Thermal data:
#
#       X = 1 ... 120
#       Y = 1 ... 90
#
THERMAL_WIDTH = 120
THERMAL_HEIGHT = 90


# ============================================================
# THERMAL REGION IN THE JPG
# ============================================================

# The JPG contains:
#
#       ┌───────────────────────────────┬─────────────┐
#       │                               │             │
#       │       THERMAL IMAGE           │ temperature │
#       │                               │    scale    │
#       │                               │             │
#       └───────────────────────────────┴─────────────┘
#
# Only the thermal-image region is used for coordinate
# conversion.
#
# Current values from your configuration:
#
#       2600 × 1920
#
# Change these if the thermal-image region in your JPGs
# has different dimensions.

THERMAL_DISPLAY_WIDTH = 2600
THERMAL_DISPLAY_HEIGHT = 1920


# ============================================================
# GLOBAL STATE
# ============================================================

points = []

markers = []
labels = []

fig = None
ax = None

image = None
IMAGE_PATH = None


# ============================================================
# LOAD EXPERIMENT FOLDER
# ============================================================

def load_routes():
    """
    Read the experiment folder and ROI image name from rutas.csv.

    Expected format:

        experiment_folder
        /path/to/experiment
        roi_image_name
        IR_00395.jpg
    """

    if not ROUTES_CSV.exists():
        raise FileNotFoundError(
            f"Routes CSV does not exist:\n"
            f"{ROUTES_CSV}"
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

    experiment_folder = None
    roi_image_name = None

    for i, value in enumerate(rows):

        if value == "experiment_folder":
            if i + 1 >= len(rows):
                raise ValueError(
                    "experiment_folder has no value in rutas.csv."
                )

            experiment_folder = rows[i + 1]

        elif value == "roi_image_name":
            if i + 1 >= len(rows):
                raise ValueError(
                    "roi_image_name has no value in rutas.csv."
                )

            roi_image_name = rows[i + 1]

    if not experiment_folder:
        raise ValueError(
            "rutas.csv must contain 'experiment_folder' "
            "followed by the experiment path."
        )

    if not roi_image_name:
        raise ValueError(
            "rutas.csv must contain 'roi_image_name' "
            "followed by the image filename."
        )

    return Path(experiment_folder), roi_image_name


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(jpg_folder, image_name):
    """
    Find the exact image specified by roi_image_name in rutas.csv.
    """

    image_path = jpg_folder / image_name

    if not image_path.exists():
        raise FileNotFoundError(
            f"ROI image not found:\n{image_path}"
        )

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"ROI image has an unsupported extension:\n"
            f"{image_path}"
        )

    return image_path


# ============================================================
# CONVERT JPG POSITION → THERMAL CSV POSITION
# ============================================================

def image_to_thermal(x, y):
    """
    Convert a position in the thermal portion of the JPG
    to the coordinate system used by the thermal CSV.

    JPG thermal region:

        x = 0 ... THERMAL_DISPLAY_WIDTH
        y = 0 ... THERMAL_DISPLAY_HEIGHT

    Thermal CSV:

        x = 1 ... 120
        y = 1 ... 90
    """

    thermal_x = int(
        x
        / THERMAL_DISPLAY_WIDTH
        * THERMAL_WIDTH
    ) + 1

    thermal_y = int(
        y
        / THERMAL_DISPLAY_HEIGHT
        * THERMAL_HEIGHT
    ) + 1

    # --------------------------------------------------------
    # Keep coordinates inside valid range
    # --------------------------------------------------------

    thermal_x = max(
        1,
        min(
            THERMAL_WIDTH,
            thermal_x
        )
    )

    thermal_y = max(
        1,
        min(
            THERMAL_HEIGHT,
            thermal_y
        )
    )

    return thermal_x, thermal_y


# ============================================================
# MOUSE CLICK
# ============================================================

def on_click(event):

    if event.inaxes != ax:
        return

    # --------------------------------------------------------
    # Only left mouse button
    # --------------------------------------------------------

    if event.button != 1:
        return

    if (
        event.xdata is None
        or event.ydata is None
    ):
        return

    x = event.xdata
    y = event.ydata

    # ========================================================
    # CHECK THERMAL REGION
    # ========================================================

    if (
        x < 0
        or x >= THERMAL_DISPLAY_WIDTH
    ):

        print(
            "Click ignored: outside thermal image."
        )

        return

    if (
        y < 0
        or y >= THERMAL_DISPLAY_HEIGHT
    ):

        print(
            "Click ignored: outside thermal image."
        )

        return

    # ========================================================
    # CONVERT TO THERMAL COORDINATES
    # ========================================================

    thermal_x, thermal_y = (
        image_to_thermal(
            x,
            y
        )
    )

    # ========================================================
    # POINT NAME
    # ========================================================

    number = len(points) + 1

    name = f"point_{number}"

    # ========================================================
    # STORE POINT
    # ========================================================

    points.append({

        "name":
            name,

        "x_coordinate":
            thermal_x,

        "y_coordinate":
            thermal_y,

        "image_x":
            x,

        "image_y":
            y
    })

    # ========================================================
    # DRAW MARKER
    # ========================================================

    marker, = ax.plot(

        x,
        y,

        marker="o",

        markersize=8,

        markerfacecolor="none",

        markeredgecolor="white",

        markeredgewidth=2
    )

    # ========================================================
    # DRAW LABEL
    # ========================================================

    label = ax.text(

        x + 10,

        y - 10,

        (
            f"{name}\n"
            f"({thermal_x}, {thermal_y})"
        ),

        color="white",

        fontsize=10,

        fontweight="bold",

        bbox=dict(

            facecolor="black",

            alpha=0.65,

            edgecolor="none",

            pad=3
        )
    )

    markers.append(
        marker
    )

    labels.append(
        label
    )

    fig.canvas.draw_idle()

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print(

        f"{name}: "

        f"JPG=({x:.1f}, {y:.1f}) "

        f"→ CSV=({thermal_x}, {thermal_y})"
    )


# ============================================================
# REDRAW LABELS
# ============================================================

def redraw_labels():

    global labels

    # --------------------------------------------------------
    # Remove current labels
    # --------------------------------------------------------

    for label in labels:

        label.remove()

    labels.clear()

    # --------------------------------------------------------
    # Recreate labels
    # --------------------------------------------------------

    for point in points:

        label = ax.text(

            point["image_x"] + 10,

            point["image_y"] - 10,

            (
                f"{point['name']}\n"
                f"({point['x_coordinate']}, "
                f"{point['y_coordinate']})"
            ),

            color="white",

            fontsize=10,

            fontweight="bold",

            bbox=dict(

                facecolor="black",

                alpha=0.65,

                edgecolor="none",

                pad=3
            )
        )

        labels.append(
            label
        )


# ============================================================
# KEYBOARD
# ============================================================

def on_key(event):

    # ========================================================
    # ENTER → SAVE
    # ========================================================

    if event.key == "enter":

        save_rois()

        plt.close(fig)

    # ========================================================
    # BACKSPACE → REMOVE LAST
    # ========================================================

    elif event.key == "backspace":

        if not points:

            print(
                "No points to remove."
            )

            return

        # ----------------------------------------------------
        # Remove last point
        # ----------------------------------------------------

        removed = points.pop()

        # ----------------------------------------------------
        # Remove last marker
        # ----------------------------------------------------

        marker = markers.pop()

        marker.remove()

        # ----------------------------------------------------
        # Rename remaining points
        # ----------------------------------------------------

        for i, point in enumerate(
            points
        ):

            point["name"] = (
                f"point_{i + 1}"
            )

        # ----------------------------------------------------
        # Redraw labels
        # ----------------------------------------------------

        redraw_labels()

        fig.canvas.draw_idle()

        print(
            f"Removed {removed['name']}."
        )

    # ========================================================
    # ESC → CANCEL
    # ========================================================

    elif event.key == "escape":

        print(
            "Selection cancelled."
        )

        points.clear()

        plt.close(fig)


# ============================================================
# SAVE ROI CSV
# ============================================================

def save_rois():

    if not points:

        print(
            "No points selected."
        )

        return

    # --------------------------------------------------------
    # Output:
    #
    # experiment/
    #     rois.csv
    #
    # --------------------------------------------------------

    experiment_folder, _ = load_routes()

    roi_csv = (
        experiment_folder
        / "rois.csv"
    )

    # --------------------------------------------------------
    # Make sure experiment folder exists
    # --------------------------------------------------------

    roi_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(

        roi_csv,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "name",
            "x_coordinate",
            "y_coordinate"
        ])

        for point in points:

            writer.writerow([

                point["name"],

                point["x_coordinate"],

                point["y_coordinate"]
            ])

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()

    print(
        "============================================"
    )

    print(
        "ROI FILE SAVED"
    )

    print(
        "============================================"
    )

    print()

    for point in points:

        print(

            f"{point['name']}: "

            f"x={point['x_coordinate']}, "

            f"y={point['y_coordinate']}"
        )

    print()

    print(
        "File:"
    )

    print(
        roi_csv
    )

    print()

    print(
        "============================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global fig
    global ax
    global image
    global IMAGE_PATH

    # ========================================================
    # LOAD EXPERIMENT FOLDER
    # ========================================================

    try:

        experiment_folder, roi_image_name = load_routes()

    except Exception as e:

        print(
            "ERROR loading rutas.csv:"
        )

        print(
            e
        )

        return

    # ========================================================
    # DEFINE FOLDERS
    # ========================================================

    jpg_folder = (
        experiment_folder
        / "jpg"
    )

    # ========================================================
    # FIND IMAGE
    # ========================================================

    try:

        IMAGE_PATH = (
            find_image(
                jpg_folder,
                roi_image_name
            )
        )

    except Exception as e:

        print(
            "ERROR finding image:"
        )

        print(
            e
        )

        return

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    try:

        image = Image.open(
            IMAGE_PATH
        )

    except Exception as e:

        print(
            "ERROR opening image:"
        )

        print(
            e
        )

        return

    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        "============================================"
    )

    print(
        "THERMAL ROI SELECTOR"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Experiment folder:"
    )

    print(
        experiment_folder
    )

    print()

    print(
        f"JPG folder:"
    )

    print(
        jpg_folder
    )

    print()

    print(
        f"ROI image from rutas.csv:"
    )

    print(
        roi_image_name
    )

    print(
        f"Image:"
    )

    print(
        IMAGE_PATH.name
    )

    print()

    print(
        f"JPG size:"
    )

    print(
        f"{image.width} × {image.height}"
    )

    print()

    print(
        f"Thermal display region:"
    )

    print(
        f"{THERMAL_DISPLAY_WIDTH} × "
        f"{THERMAL_DISPLAY_HEIGHT}"
    )

    print()

    print(
        f"Thermal CSV:"
    )

    print(
        f"{THERMAL_WIDTH} × "
        f"{THERMAL_HEIGHT}"
    )

    print()

    # ========================================================
    # CONTROLS
    # ========================================================

    print(
        "Controls:"
    )

    print()

    print(
        "  LEFT CLICK  → select point"
    )

    print(
        "  BACKSPACE   → remove last point"
    )

    print(
        "  ENTER       → save rois.csv"
    )

    print(
        "  ESC         → cancel"
    )

    print()

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(15, 10)
    )

    ax.imshow(
        image
    )

    # ========================================================
    # SHOW THERMAL REGION BOUNDARY
    # ========================================================

    ax.axvline(

        THERMAL_DISPLAY_WIDTH,

        color="cyan",

        linewidth=2,

        linestyle="--"
    )

    ax.text(

        THERMAL_DISPLAY_WIDTH - 10,

        30,

        "THERMAL DATA",

        color="white",

        fontsize=12,

        fontweight="bold",

        horizontalalignment="right",

        bbox=dict(

            facecolor="black",

            alpha=0.6,

            edgecolor="none"
        )
    )

    # ========================================================
    # PLOT LIMITS
    # ========================================================

    ax.set_xlim(

        0,

        image.width
    )

    ax.set_ylim(

        image.height,

        0
    )

    ax.set_xlabel(
        "JPG X coordinate"
    )

    ax.set_ylabel(
        "JPG Y coordinate"
    )

    ax.set_title(

        "Click points — ENTER to save — "
        "BACKSPACE to undo — ESC to cancel"
    )

    # ========================================================
    # CONNECT MOUSE EVENT
    # ========================================================

    fig.canvas.mpl_connect(

        "button_press_event",

        on_click
    )

    # ========================================================
    # CONNECT KEYBOARD EVENT
    # ========================================================

    fig.canvas.mpl_connect(

        "key_press_event",

        on_key
    )

    # ========================================================
    # SHOW
    # ========================================================

    plt.tight_layout()

    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
