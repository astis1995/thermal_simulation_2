from pathlib import Path
import re

# -----------------------------------------------------------------------------
# Pattern:
# IR_00085_7-17-26_02.38.59.jpg
# IR_00085_7-17-26_02.30.12.csv
#
# Code = IR_00085
# Timestamp = 7-17-26_02.38.59
# -----------------------------------------------------------------------------

image_pattern = re.compile(
    r"^(?P<code>IR_\d+)_(?P<timestamp>\d+-\d+-\d+_\d+\.\d+\.\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

csv_pattern = re.compile(
    r"^(?P<code>IR_\d+)_.*\.csv$",
    re.IGNORECASE,
)

def synchronize_csv_names(folder):
    folder = Path(folder)

    images = {}
    csvs = {}

    # Collect files
    for f in folder.iterdir():

        if f.suffix.lower() == ".csv":
            m = csv_pattern.match(f.name)
            if m:
                csvs[m.group("code")] = f

        elif f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = image_pattern.match(f.name)
            if m:
                images[m.group("code")] = (m.group("timestamp"), f)

    # Rename CSVs if needed
    for code, (img_timestamp, img_file) in images.items():

        if code not in csvs:
            print(f"No CSV found for {code}")
            continue

        csv_file = csvs[code]

        new_name = f"{code}_{img_timestamp}.csv"
        new_path = csv_file.with_name(new_name)

        if csv_file.name == new_name:
            print(f"OK: {csv_file.name}")
            continue

        print(f"Renaming:")
        print(f"  {csv_file.name}")
        print(f"-> {new_name}")

        csv_file.rename(new_path)


if __name__ == "__main__":
    folder = input("Folder: ").strip().strip('"')
    synchronize_csv_names(folder)
