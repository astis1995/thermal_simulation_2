import csv
import re
from pathlib import Path
from datetime import datetime, timedelta

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CSV containing the experiment folder
#
# Example:
#
# experiment_folder
# /mnt/c/Users/esteb/Downloads/output/output/experimento2
#
# ------------------------------------------------------------

ROUTES_CSV = Path(
    r"rutas.csv"
)


# ------------------------------------------------------------
# Image extensions to process
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
# LOAD EXPERIMENT FOLDER
# ============================================================

def load_experiment_folder():
    """
    Read the experiment root folder from rutas.csv.
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
            "experiment_folder in rutas.csv is empty."
        )

    return Path(experiment_folder)


# ============================================================
# EXTRACT IMAGE NUMBER
# ============================================================

def get_image_number(filename):
    """
    Extract numerical image ID from filenames such as:

        IR_00327.jpg
        IR_00328.jpg
        IR_10001.jpg

    Returns:

        327
        328
        10001

    """

    match = re.search(
        r"(\d+)(?=\.[^.]+$)",
        filename
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_ocr_from_roi(image_path):
    """
    Extract OCR from:

        Horizontal: 25% - 75%
        Vertical:   bottom 20% (80% - 100%)
    """

    image = Image.open(
        image_path
    )

    width, height = image.size

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    x1 = int(
        width * 0.25
    )

    x2 = int(
        width * 0.75
    )

    y1 = int(
        height * 0.80
    )

    y2 = height

    roi = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    roi = roi.convert("L")

    # --------------------------------------------------------
    # Enlarge image for OCR
    # --------------------------------------------------------

    scale = 3

    roi = roi.resize(
        (
            roi.width * scale,
            roi.height * scale
        )
    )

    # --------------------------------------------------------
    # Increase contrast
    # --------------------------------------------------------

    roi = ImageEnhance.Contrast(
        roi
    ).enhance(2.0)

    # --------------------------------------------------------
    # Slight sharpening
    # --------------------------------------------------------

    roi = roi.filter(
        ImageFilter.SHARPEN
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        roi,
        config="--psm 6"
    )

    # --------------------------------------------------------
    # Clean whitespace
    # --------------------------------------------------------

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# EXTRACT / FIX TIMESTAMP
# ============================================================

def fix_timestamp(text):
    """
    Extract a valid timestamp from OCR text.

    Garbage before and after the timestamp is discarded.

    Examples:

        : : 9/4/2026 1:56:02 PM

        9/4/2026 2:04:34 PM 8

        | 9/4/2026 2:07:39 PM 18

        9/4/2026 2:07:58 PM 178

    Result:

        9/4/2026 2:07:58 PM
    """

    if not text:

        return None

    # --------------------------------------------------------
    # Common OCR character corrections
    # --------------------------------------------------------

    corrections = {

        "O": "0",
        "o": "0",

        "I": "1",
        "l": "1",

        "|": "1",
    }

    for old, new in corrections.items():

        text = text.replace(
            old,
            new
        )

    # --------------------------------------------------------
    # Timestamp pattern
    # --------------------------------------------------------

    pattern = re.compile(
        r"(\d{1,2})\s*/\s*"
        r"(\d{1,2})\s*/\s*"
        r"(\d{4})\s+"
        r"(\d{1,2})\s*:\s*"
        r"(\d{2})\s*:\s*"
        r"(\d{2})\s*"
        r"([APap])\s*[Mm]"
    )

    match = pattern.search(
        text
    )

    if not match:

        return None

    (
        month,
        day,
        year,
        hour,
        minute,
        second,
        ampm
    ) = match.groups()

    # --------------------------------------------------------
    # Convert numeric values
    # --------------------------------------------------------

    month = int(month)
    day = int(day)
    year = int(year)

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    ampm = (
        ampm.upper()
        + "M"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not 1 <= month <= 12:

        return None

    if not 1 <= day <= 31:

        return None

    if not 1 <= hour <= 12:

        return None

    if not 0 <= minute <= 59:

        return None

    if not 0 <= second <= 59:

        return None

    # --------------------------------------------------------
    # Convert to datetime
    # --------------------------------------------------------

    try:

        dt = datetime.strptime(
            f"{month}/{day}/{year} "
            f"{hour}:{minute}:{second} "
            f"{ampm}",

            "%m/%d/%Y %I:%M:%S %p"
        )

        return dt

    except ValueError:

        return None


# ============================================================
# FORMAT TIMESTAMP
# ============================================================

def format_timestamp(dt):

    if dt is None:

        return ""

    # Cross-platform formatting
    # Avoid %-I because it is not supported on Windows.

    return dt.strftime(
        "%m/%d/%Y %I:%M:%S %p"
    ).replace(
        " 0",
        " "
    )


# ============================================================
# REPAIR TIMESTAMP SEQUENCE
# ============================================================

def repair_timestamp_sequence(results):
    """
    Ensures that timestamps increase with image number.

    If:

        image 327 -> 1:56:02
        image 328 -> 1:55:59
        image 329 -> 1:56:21

    the middle timestamp is considered invalid.

    If there are valid timestamps before and after it,
    the missing/invalid timestamp is placed between them.

    If there is no future timestamp, one second is added
    to the previous valid timestamp.
    """

    corrections = 0

    for i in range(
        len(results)
    ):

        current = results[i][
            "timestamp_dt"
        ]

        previous = None
        next_valid = None

        # ----------------------------------------------------
        # Find previous valid timestamp
        # ----------------------------------------------------

        for j in range(
            i - 1,
            -1,
            -1
        ):

            if results[j][
                "timestamp_dt"
            ] is not None:

                previous = results[j][
                    "timestamp_dt"
                ]

                break

        # ----------------------------------------------------
        # Find next valid timestamp
        # ----------------------------------------------------

        for j in range(
            i + 1,
            len(results)
        ):

            if results[j][
                "timestamp_dt"
            ] is not None:

                next_valid = results[j][
                    "timestamp_dt"
                ]

                break

        # ====================================================
        # Missing timestamp
        # ====================================================

        if current is None:

            # ------------------------------------------------
            # Previous AND next timestamp available
            # ------------------------------------------------

            if (
                previous is not None
                and
                next_valid is not None
            ):

                gap = (
                    next_valid
                    - previous
                ).total_seconds()

                if gap > 1:

                    new_time = (
                        previous
                        + timedelta(
                            seconds=max(
                                1,
                                int(
                                    gap / 2
                                )
                            )
                        )
                    )

                    if new_time >= next_valid:

                        new_time = (
                            next_valid
                            - timedelta(
                                seconds=1
                            )
                        )

                    results[i][
                        "timestamp_dt"
                    ] = new_time

                    results[i][
                        "corrected"
                    ] = True

                    corrections += 1

            # ------------------------------------------------
            # Only previous timestamp available
            # ------------------------------------------------

            elif previous is not None:

                results[i][
                    "timestamp_dt"
                ] = (
                    previous
                    + timedelta(
                        seconds=1
                    )
                )

                results[i][
                    "corrected"
                ] = True

                corrections += 1

            continue

        # ====================================================
        # Timestamp goes backwards or is equal
        # ====================================================

        if (
            previous is not None
            and
            current <= previous
        ):

            # ------------------------------------------------
            # Previous AND next timestamp available
            # ------------------------------------------------

            if (
                next_valid is not None
                and
                next_valid > previous
            ):

                gap = (
                    next_valid
                    - previous
                ).total_seconds()

                if gap > 1:

                    new_time = (
                        previous
                        + timedelta(
                            seconds=max(
                                1,
                                int(
                                    gap / 2
                                )
                            )
                        )
                    )

                    if new_time >= next_valid:

                        new_time = (
                            next_valid
                            - timedelta(
                                seconds=1
                            )
                        )

                    results[i][
                        "timestamp_dt"
                    ] = new_time

                    results[i][
                        "corrected"
                    ] = True

                    corrections += 1

            # ------------------------------------------------
            # No future timestamp
            # ------------------------------------------------

            else:

                results[i][
                    "timestamp_dt"
                ] = (
                    previous
                    + timedelta(
                        seconds=1
                    )
                )

                results[i][
                    "corrected"
                ] = True

                corrections += 1

    return corrections


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================"
    )

    print(
        "THERMAL TIMESTAMP OCR"
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
            f"ERROR loading rutas.csv:\n{e}"
        )

        return

    # --------------------------------------------------------
    # Folder structure
    # --------------------------------------------------------

    jpg_folder = (
        experiment_folder
        / "jpg"
    )

    output_csv = (
        experiment_folder
        / "thermal_timestamps.csv"
    )

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
        f"Output:"
    )

    print(
        output_csv
    )

    print()

    # ========================================================
    # CHECK JPG FOLDER
    # ========================================================

    if not experiment_folder.exists():

        print(
            "ERROR: Experiment folder does not exist:"
        )

        print(
            experiment_folder
        )

        return

    if not jpg_folder.exists():

        print(
            "ERROR: JPG folder does not exist:"
        )

        print(
            jpg_folder
        )

        return

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_files = sorted(
        [
            f
            for f in jpg_folder.iterdir()
            if (
                f.is_file()
                and
                f.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda f: (
            get_image_number(
                f.name
            )
            if get_image_number(
                f.name
            ) is not None
            else float("inf")
        )
    )

    print(
        f"Found {len(image_files)} images."
    )

    print()

    if not image_files:

        print(
            "No images found."
        )

        return

    results = []

    # ========================================================
    # PASS 1: OCR
    # ========================================================

    print(
        "============================================"
    )

    print(
        "PASS 1: OCR"
    )

    print(
        "============================================"
    )

    print()

    for i, image_path in enumerate(
        image_files,
        start=1
    ):

        image_number = (
            get_image_number(
                image_path.name
            )
        )

        print(
            f"[{i}/{len(image_files)}] "
            f"{image_path.name}"
        )

        try:

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            raw_text = (
                extract_ocr_from_roi(
                    image_path
                )
            )

            # ------------------------------------------------
            # Extract timestamp
            # ------------------------------------------------

            timestamp = (
                fix_timestamp(
                    raw_text
                )
            )

            print(
                f"    Raw OCR: {raw_text}"
            )

            if timestamp is not None:

                print(
                    "    Timestamp: "
                    f"{format_timestamp(timestamp)}"
                )

            else:

                print(
                    "    Timestamp: "
                    "NOT FOUND"
                )

            results.append({

                "filename":
                    image_path.name,

                "image_number":
                    image_number,

                "ocr_raw":
                    raw_text,

                "timestamp_dt":
                    timestamp,

                "corrected":
                    False
            })

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

            results.append({

                "filename":
                    image_path.name,

                "image_number":
                    image_number,

                "ocr_raw":
                    "",

                "timestamp_dt":
                    None,

                "corrected":
                    False
            })

        print()

    # ========================================================
    # PASS 2: TIMESTAMP SEQUENCE
    # ========================================================

    print(
        "============================================"
    )

    print(
        "PASS 2: CHECKING TIMESTAMP SEQUENCE"
    )

    print(
        "============================================"
    )

    print()

    corrections = (
        repair_timestamp_sequence(
            results
        )
    )

    print(
        f"Timestamp corrections: "
        f"{corrections}"
    )

    print()

    # ========================================================
    # CALCULATE ELAPSED TIME
    # ========================================================

    first_timestamp = None

    for row in results:

        if row[
            "timestamp_dt"
        ] is not None:

            first_timestamp = row[
                "timestamp_dt"
            ]

            break

    for row in results:

        timestamp = row[
            "timestamp_dt"
        ]

        if (
            timestamp is not None
            and
            first_timestamp is not None
        ):

            elapsed = (
                timestamp
                - first_timestamp
            ).total_seconds()

        else:

            elapsed = ""

        row[
            "elapsed_seconds"
        ] = elapsed

    # ========================================================
    # SAVE CSV
    # ========================================================

    print(
        "============================================"
    )

    print(
        "SAVING CSV"
    )

    print(
        "============================================"
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filename",
                "image_number",
                "ocr_raw",
                "timestamp",
                "elapsed_seconds",
                "corrected"
            ]
        )

        writer.writeheader()

        for row in results:

            writer.writerow({

                "filename":
                    row[
                        "filename"
                    ],

                "image_number":
                    row[
                        "image_number"
                    ],

                "ocr_raw":
                    row[
                        "ocr_raw"
                    ],

                "timestamp":
                    format_timestamp(
                        row[
                            "timestamp_dt"
                        ]
                    ),

                "elapsed_seconds":
                    row[
                        "elapsed_seconds"
                    ],

                "corrected":
                    row[
                        "corrected"
                    ]
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

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
        f"Images processed: "
        f"{len(results)}"
    )

    print(
        f"Timestamp corrections: "
        f"{corrections}"
    )

    print()

    print(
        "CSV saved to:"
    )

    print(
        output_csv
    )

    print(
        "============================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
