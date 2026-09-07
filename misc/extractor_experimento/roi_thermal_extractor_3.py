import csv
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


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
# ------------------------------------------------------------

ROUTES_CSV = Path(
    r"rutas.csv"
)


# ============================================================
# LOAD EXPERIMENT FOLDER
# ============================================================

def load_experiment_folder():
    """
    Read the experiment root folder from rutas.csv.

    Expected format:

        experiment_folder
        /path/to/experiment
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

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise ValueError(
                "rutas.csv has no header."
            )

        if "experiment_folder" not in reader.fieldnames:

            raise ValueError(
                "rutas.csv must contain the column:\n"
                "experiment_folder"
            )

        try:

            row = next(reader)

        except StopIteration:

            raise ValueError(
                "rutas.csv is empty."
            )

    experiment_folder = row[
        "experiment_folder"
    ].strip()

    if not experiment_folder:

        raise ValueError(
            "experiment_folder is empty in rutas.csv."
        )

    return Path(
        experiment_folder
    )


# ============================================================
# LOAD ROIS
# ============================================================

def load_rois(roi_path):
    """
    Load points of interest.

    Expected format:

        name,x_coordinate,y_coordinate

    Example:

        Center,60,45
        Left,20,45
        Right,100,45

    Coordinates are 1-based and match the thermal CSV.
    """

    rois = []

    with open(
        roi_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "name",
            "x_coordinate",
            "y_coordinate"
        }

        if (
            reader.fieldnames is None
            or
            not required_columns.issubset(
                reader.fieldnames
            )
        ):

            raise ValueError(
                "ROI CSV must contain columns:\n"
                "name,x_coordinate,y_coordinate"
            )

        for row in reader:

            name = row[
                "name"
            ].strip()

            x = int(
                row[
                    "x_coordinate"
                ]
            )

            y = int(
                row[
                    "y_coordinate"
                ]
            )

            rois.append({

                "name": name,

                "x": x,

                "y": y
            })

    return rois


# ============================================================
# LOAD TIMESTAMPS
# ============================================================

def load_timestamps(timestamp_path):
    """
    Load timestamps from thermal_timestamps.csv.

    Expected columns:

        filename
        image_number
        ocr_raw
        timestamp
        elapsed_seconds
        corrected

    The extension is removed so:

        IR_00327.jpg

    matches:

        IR_00327.csv
    """

    timestamps = {}

    with open(
        timestamp_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise ValueError(
                "Timestamp CSV has no header."
            )

        if "filename" not in reader.fieldnames:

            raise ValueError(
                "Timestamp CSV must contain "
                "a 'filename' column."
            )

        if "timestamp" not in reader.fieldnames:

            raise ValueError(
                "Timestamp CSV must contain "
                "a 'timestamp' column."
            )

        for row in reader:

            filename = row[
                "filename"
            ].strip()

            if not filename:

                continue

            timestamp_text = row[
                "timestamp"
            ].strip()

            if not timestamp_text:

                continue

            # ------------------------------------------------
            # Remove extension
            # ------------------------------------------------

            stem = Path(
                filename
            ).stem

            try:

                timestamp = datetime.strptime(

                    timestamp_text,

                    "%m/%d/%Y %I:%M:%S %p"
                )

                timestamps[
                    stem
                ] = timestamp

            except ValueError:

                print(

                    f"WARNING: Could not parse "
                    f"timestamp for {filename}: "
                    f"{timestamp_text}"
                )

    return timestamps


# ============================================================
# READ THERMAL CSV
# ============================================================

def read_thermal_csv(csv_path):
    """
    Read a Fluke thermal CSV.

    Structure:

        Lines 1-4 = metadata
        Line 5    = X coordinates
        Lines 6-95 = thermal matrix

    Returns a pandas DataFrame:

        index   = Y coordinate
        columns = X coordinate
        values  = temperature in °C
    """

    # --------------------------------------------------------
    # Read UTF-16 file
    # --------------------------------------------------------

    with open(
        csv_path,
        "r",
        encoding="utf-16"
    ) as f:

        lines = f.readlines()

    # --------------------------------------------------------
    # Check minimum number of lines
    # --------------------------------------------------------

    if len(lines) < 95:

        raise ValueError(
            f"{csv_path.name} has only "
            f"{len(lines)} lines. "
            f"Expected at least 95."
        )

    # ========================================================
    # HEADER
    # ========================================================

    # Python index 4 = line 5

    header = lines[
        4
    ].strip().split(",")

    if (
        header
        and
        header[-1] == ""
    ):

        header = header[:-1]

    # --------------------------------------------------------
    # First field is Y label.
    # Remaining fields are X coordinates.
    # --------------------------------------------------------

    x_coordinates = [

        int(value)

        for value in header[1:]

        if value.strip()
    ]

    # ========================================================
    # THERMAL DATA
    # ========================================================

    # Lines 6-95 inclusive
    #
    # Python:
    #
    # lines[5:95]

    matrix = []

    y_coordinates = []

    for line in lines[
        5:95
    ]:

        line = line.strip()

        if not line:

            continue

        values = line.split(",")

        if (
            values
            and
            values[-1] == ""
        ):

            values = values[:-1]

        if len(values) < 2:

            continue

        # ----------------------------------------------------
        # Y coordinate
        # ----------------------------------------------------

        try:

            y = int(
                values[0]
            )

        except ValueError:

            continue

        # ----------------------------------------------------
        # Temperature values
        # ----------------------------------------------------

        temperatures = []

        for value in values[1:]:

            try:

                temperatures.append(
                    float(value)
                )

            except ValueError:

                temperatures.append(
                    float("nan")
                )

        # ----------------------------------------------------
        # Validate row size
        # ----------------------------------------------------

        if (
            len(temperatures)
            !=
            len(x_coordinates)
        ):

            print(

                f"WARNING: {csv_path.name}: "
                f"row {y} has "
                f"{len(temperatures)} values, "
                f"expected "
                f"{len(x_coordinates)}"
            )

            continue

        y_coordinates.append(
            y
        )

        matrix.append(
            temperatures
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(

        matrix,

        index=y_coordinates,

        columns=x_coordinates
    )

    df.index.name = "y"

    return df


# ============================================================
# EXTRACT POINTS
# ============================================================

def extract_points(df, rois):
    """
    Extract all requested pixels from a thermal DataFrame.
    """

    values = {}

    for roi in rois:

        name = roi[
            "name"
        ]

        x = roi[
            "x"
        ]

        y = roi[
            "y"
        ]

        # ----------------------------------------------------
        # Check Y coordinate
        # ----------------------------------------------------

        if y not in df.index:

            raise ValueError(

                f"Y coordinate {y} "
                f"does not exist."
            )

        # ----------------------------------------------------
        # Check X coordinate
        # ----------------------------------------------------

        if x not in df.columns:

            raise ValueError(

                f"X coordinate {x} "
                f"does not exist."
            )

        # ----------------------------------------------------
        # Extract temperature
        # ----------------------------------------------------

        temperature = df.loc[
            y,
            x
        ]

        values[
            name
        ] = temperature

    return values


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================"
    )

    print(
        "THERMAL TIME-SERIES EXTRACTION"
    )

    print(
        "============================================"
    )

    print()

    # ========================================================
    # LOAD EXPERIMENT FOLDER
    # ========================================================

    try:

        experiment_folder = (
            load_experiment_folder()
        )

    except Exception as e:

        print(
            "ERROR loading rutas.csv:"
        )

        print(
            e
        )

        return

    # ========================================================
    # DEFINE PATHS
    # ========================================================

    # Original thermal CSV files
    THERMAL_FOLDER = (
        experiment_folder
        / "csv"
    )

    # Timestamp intermediate CSV
    TIMESTAMP_CSV = (
        experiment_folder
        / "thermal_timestamps.csv"
    )

    # ROI intermediate CSV
    ROI_CSV = (
        experiment_folder
        / "rois.csv"
    )

    # Output folder
    OUTPUT_FOLDER = (
        experiment_folder
        / "output"
    )

    # Final result
    OUTPUT_CSV = (
        OUTPUT_FOLDER
        / "thermal_time_series.csv"
    )

    # ========================================================
    # CREATE OUTPUT FOLDER
    # ========================================================

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DISPLAY PATHS
    # ========================================================

    print(
        "Experiment folder:"
    )

    print(
        experiment_folder
    )

    print()

    print(
        "Thermal CSV folder:"
    )

    print(
        THERMAL_FOLDER
    )

    print()

    print(
        "Timestamp CSV:"
    )

    print(
        TIMESTAMP_CSV
    )

    print()

    print(
        "ROI CSV:"
    )

    print(
        ROI_CSV
    )

    print()

    print(
        "Output CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print()

    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not experiment_folder.exists():

        raise FileNotFoundError(

            f"Experiment folder does not exist:\n"
            f"{experiment_folder}"
        )

    if not THERMAL_FOLDER.exists():

        raise FileNotFoundError(

            f"Thermal folder does not exist:\n"
            f"{THERMAL_FOLDER}"
        )

    if not TIMESTAMP_CSV.exists():

        raise FileNotFoundError(

            f"Timestamp CSV does not exist:\n"
            f"{TIMESTAMP_CSV}"
        )

    if not ROI_CSV.exists():

        raise FileNotFoundError(

            f"ROI CSV does not exist:\n"
            f"{ROI_CSV}"
        )

    # ========================================================
    # LOAD ROIS
    # ========================================================

    rois = load_rois(
        ROI_CSV
    )

    print(
        f"Loaded {len(rois)} "
        f"points of interest:"
    )

    print()

    for roi in rois:

        print(

            f"  {roi['name']}: "
            f"x={roi['x']}, "
            f"y={roi['y']}"
        )

    print()

    # ========================================================
    # LOAD TIMESTAMPS
    # ========================================================

    timestamps = load_timestamps(
        TIMESTAMP_CSV
    )

    print(
        f"Loaded {len(timestamps)} "
        f"timestamps."
    )

    print()

    # ========================================================
    # FIND THERMAL CSV FILES
    # ========================================================

    thermal_files = sorted(

        [

            f

            for f in THERMAL_FOLDER.iterdir()

            if (

                f.is_file()

                and

                f.suffix.lower() == ".csv"

            )
        ],

        key=lambda f: (

            int(
                re.search(
                    r"\d+",
                    f.stem
                ).group()
            )

            if re.search(
                r"\d+",
                f.stem
            )

            else float("inf")
        )
    )

    print(
        f"Found {len(thermal_files)} "
        f"thermal CSV files."
    )

    print()

    if not thermal_files:

        print(
            "ERROR: No thermal CSV files found."
        )

        return

    # ========================================================
    # PROCESS FILES
    # ========================================================

    results = []

    for i, thermal_file in enumerate(

        thermal_files,

        start=1
    ):

        print(

            f"[{i}/{len(thermal_files)}] "
            f"{thermal_file.name}"
        )

        # ----------------------------------------------------
        # File identifier
        # ----------------------------------------------------

        stem = thermal_file.stem

        # ----------------------------------------------------
        # Find timestamp
        # ----------------------------------------------------

        timestamp = timestamps.get(
            stem
        )

        if timestamp is None:

            print(
                "    WARNING: "
                "No timestamp found."
            )

            print()

            continue

        print(

            f"    Timestamp: "
            f"{timestamp.strftime('%m/%d/%Y %I:%M:%S %p')}"
        )

        # ----------------------------------------------------
        # Read thermal matrix
        # ----------------------------------------------------

        try:

            df = read_thermal_csv(
                thermal_file
            )

        except Exception as e:

            print(

                f"    ERROR reading "
                f"thermal CSV: {e}"
            )

            print()

            continue

        print(

            f"    Matrix: "
            f"{len(df.columns)} x "
            f"{len(df.index)}"
        )

        # ----------------------------------------------------
        # Extract ROIs
        # ----------------------------------------------------

        try:

            point_values = (
                extract_points(
                    df,
                    rois
                )
            )

        except Exception as e:

            print(

                f"    ERROR extracting "
                f"points: {e}"
            )

            print()

            continue

        # ----------------------------------------------------
        # Create result row
        # ----------------------------------------------------

        row = {

            "time":
                timestamp
        }

        row.update(
            point_values
        )

        results.append(
            row
        )

        # ----------------------------------------------------
        # Display values
        # ----------------------------------------------------

        for roi in rois:

            name = roi[
                "name"
            ]

            print(

                f"    {name}: "
                f"{point_values[name]:.2f} °C"
            )

        print()

    # ========================================================
    # CREATE FINAL DATAFRAME
    # ========================================================

    if not results:

        print(
            "ERROR: No valid data "
            "was extracted."
        )

        return

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    result_df = (
        result_df
        .sort_values(
            by="time"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # FORMAT TIME
    # ========================================================

    result_df["time"] = (
        result_df["time"]
        .dt.strftime(
            "%m/%d/%Y %I:%M:%S %p"
        )
    )

    # ========================================================
    # SAVE FINAL CSV
    # ========================================================

    result_df.to_csv(

        OUTPUT_CSV,

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "============================================"
    )

    print(
        "FINISHED"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Thermal files processed: "
        f"{len(results)}"
    )

    print(
        f"Points extracted: "
        f"{len(rois)}"
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_CSV
    )

    print()

    print(
        "============================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
