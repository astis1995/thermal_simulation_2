"""
io_utils.py

Input/output utilities.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import pytesseract

@dataclass
class IRFile:
    number: int
    stem: str
    csv: Path
    jpg: Path | None


def find_ir_files(folder="", first_image=""):
    """
    Find all IR files in a folder.

    Parameters
    ----------
    folder : str
        Folder containing the IR images.

    first_image : str
        Example:
            IR_00027
        or
            IR_00027.csv
        or
            IR_00027.jpg

        If empty, processing starts from the smallest IR number.

    Returns
    -------
    list[IRFile]
    """

    # ----------------------------------------------------------
    # Select folder
    # ----------------------------------------------------------

    if folder == "":
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()

        folder = filedialog.askdirectory(title="Select experiment folder")

        if folder == "":
            raise RuntimeError("No folder selected.")

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(folder)

    # ----------------------------------------------------------
    # Find CSV files
    # ----------------------------------------------------------

    pattern = re.compile(r"IR_(\d+)\.csv$", re.IGNORECASE)

    files = []

    for csv_file in folder.glob("IR_*.csv"):

        match = pattern.match(csv_file.name)

        if match is None:
            continue

        number = int(match.group(1))

        stem = csv_file.stem

        jpg = folder / f"{stem}.jpg"

        if not jpg.exists():
            jpg = None

        files.append(
            IRFile(
                number=number,
                stem=stem,
                csv=csv_file,
                jpg=jpg,
            )
        )

    if len(files) == 0:
        raise RuntimeError("No IR CSV files were found.")

    # ----------------------------------------------------------
    # Sort numerically
    # ----------------------------------------------------------

    files.sort(key=lambda f: f.number)

    # ----------------------------------------------------------
    # Apply first image
    # ----------------------------------------------------------

    if first_image:

        first_image = Path(first_image).stem.upper()

        start = None

        for i, f in enumerate(files):

            if f.stem.upper() == first_image:
                start = i
                break

        if start is None:
            raise ValueError(
                f"{first_image} was not found in {folder}"
            )

        files = files[start:]

    return files


import numpy as np
from datetime import datetime
import pandas as pd


def load_temperature_csv(csv_file):
    """
    Load one Fluke IR CSV.

    Parameters
    ----------
    csv_file : Path

    Returns
    -------
    temperatures : ndarray
        Temperature matrix in Celsius.

    metadata : dict
        Empty dictionary (reserved for future use).
    """

    metadata = {}

    df = pd.read_csv(
        csv_file,
        encoding="utf-16",
        skiprows=3,
        header=0,
        index_col=0,      # First column contains row numbers
    )

    # Remove any trailing empty columns
    df = df.dropna(axis=1, how="all")

    # Ensure all remaining values are numeric
    temperatures = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    return temperatures, metadata

def _timestamp_from_csv(metadata):
    """
    Try to extract a timestamp from the CSV metadata.
    """

    keys = {k.lower(): v for k, v in metadata.items()}

    date = None
    time = None

    # Common field names

    for k in ["date", "acquisition_date", "capture_date"]:
        if k in keys:
            date = keys[k]
            break

    for k in ["time", "acquisition_time", "capture_time"]:
        if k in keys:
            time = keys[k]
            break

    if date is None or time is None:
        return None

    text = f"{date} {time}"

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]

    for fmt in formats:

        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    return None

from pathlib import Path
from datetime import datetime


def _timestamp_from_jpg(jpg_file):
    """
    Extract timestamp from the JPG filename.

    Expected filename format:
        IR_00124_7-17-26_02.56.59.jpg

    Date format:
        MM-DD-YY_HH.MM.SS
    """

    if jpg_file is None:
        return None

    try:
        filename = Path(jpg_file).stem  # IR_00124_7-17-26_02.56.59

        # Split from the right to handle prefixes of varying length
        _, date_str, time_str = filename.rsplit("_", 2)

        return datetime.strptime(
            f"{date_str}_{time_str}",
            "%m-%d-%y_%H.%M.%S"
        )

    except Exception:
        return None

from PIL import Image


def load_image(image_file):
    """
    Load a JPG image.

    Parameters
    ----------
    image_file : str or Path

    Returns
    -------
    PIL.Image.Image
    """

    return Image.open(image_file)

from pathlib import Path
import re
from datetime import datetime

import pytesseract
from PIL import Image


def _timestamp_from_ocr(jpg_file):

    if jpg_file is None:
        return None

    img = Image.open(jpg_file)

    w, h = img.size

    crop = img.crop((
        int(w * 0.30),
        int(h * 0.88),
        int(w * 0.70),
        h,
    ))

    # Enlarge first
    crop = crop.resize(
        (crop.width * 4, crop.height * 4),
        Image.Resampling.LANCZOS,
    )

    # Convert to grayscale
    gray = crop.convert("L")

    # Keep only nearly-white pixels
    threshold = 220

    binary = gray.point(
        lambda p: 255 if p >= threshold else 0,
        mode="L",
    )

    binary.save("ocr_binary.png")

    config = (
        "--oem 3 "
        "--psm 7 "
        "-c tessedit_char_whitelist=0123456789/:APM "
    )

    text = pytesseract.image_to_string(
        binary,
        config=config,
    )

    print(repr(text))

    text = text.replace("\n", "").strip()
    # "/" after the day recognized as "1"
    text = re.sub(
        r'(\d{1,2}/\d{1,2})1(\d{4})',
        r'\1/\2',
        text,
    )

    # Remove spaces
    text = text.replace(" ", "")

    # Remove a spurious 9 immediately after a colon
    # Examples:
    #   2:559:958PM -> 2:55:58PM
    #   2:519:29PM  -> 2:51:29PM

    def fix_time_field(match):
        field = match.group(1)

        # OCR sometimes appends a spurious 9
        # 559 -> 55
        # 958 -> 58
        if len(field) == 3 and field.endswith("9"):
            field = field[:2]
        elif len(field) == 3 and field.startswith("9"):
            field = field[1:]

        return ":" + field

    text = re.sub(r":(\d{3})", fix_time_field, text)
    # --------------------------------------------------
    # Fix common OCR mistakes
    # --------------------------------------------------

    # "/" after the day recognized as "1"
    # Example:
    #   7/17120262:52:29PM
    # becomes
    #   7/17/20262:52:29PM
    text = re.sub(
        r'(\d{1,2}/\d{1,2})1(\d{4})',
        r'\1/\2',
        text,
    )

    print(repr(text))

    match = re.search(
        r'(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}):(\d{2}):(\d{2})\s*([AP]M)',
        text,
    )

    if match:

        date = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3))
        second = int(match.group(4))
        ampm = match.group(5)

        # OCR sometimes reads 5 as 9 in the minutes
        if minute > 59:
            minute -= 40

        timestamp = (
            f"{date} "
            f"{hour}:{minute:02d}:{second:02d} "
            f"{ampm}"
        )

        return datetime.strptime(
            timestamp,
            "%m/%d/%Y %I:%M:%S %p",
        )

    return None
from datetime import datetime


def get_timestamp(ir_file, metadata=None, priority=None):
    """
    Return the acquisition timestamp using the requested priority.
    """

    if priority is None:
        priority = ["ocr", "file"]

    for source in priority:

        if source == "ocr":
            ts = _timestamp_from_ocr(ir_file.jpg)
            if ts is not None:
                return ts

        elif source == "file":
            return datetime.fromtimestamp(ir_file.csv.stat().st_mtime)

    raise RuntimeError("Could not determine timestamp.")
