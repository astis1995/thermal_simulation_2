#!/usr/bin/env python3

# Example:
# python animation_gif.py outputs/simulation2/results
#
# With custom time per frame:
# python animation_gif.py outputs/simulation2/results 0.3

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_timestamp_gif(
    folder_path,
    output_name="animation-timestamp.gif",
    frame_duration=100,
    dt_per_frame=0.1,
):
    """
    Creates an animated GIF from numbered image files.

    Accepted filenames include:

        metal.0000.png
        temperature.0012.png
        animation1.0045.png

        IR_00364.jpg
        IR_00365.jpg
        IR_00366.jpg

        frame_0001.jpeg

    The GIF is written to the same directory.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder.resolve()}"
        )

    # --------------------------------------------------------
    # Allow passing the simulation folder instead of results
    # --------------------------------------------------------

    if (folder / "results").is_dir():
        folder = folder / "results"

    # --------------------------------------------------------
    # Match numbered image files
    #
    # Accepted:
    #
    #   name.0001.png
    #   name_0001.png
    #   name.0001.jpg
    #   name_0001.jpg
    #
    # The separator can be "." or "_".
    # --------------------------------------------------------

    pattern = re.compile(
        r"^(.+?)[._](\d+)\.(jpg|jpeg|png)$",
        re.IGNORECASE,
    )

    frames = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        match = pattern.match(file.name)

        if match:

            prefix = match.group(1)
            frame_number = int(match.group(2))

            frames.append(
                (
                    frame_number,
                    prefix,
                    file,
                )
            )

    # --------------------------------------------------------
    # Check that frames were found
    # --------------------------------------------------------

    if not frames:
        raise RuntimeError(
            f"No numbered JPG/JPEG/PNG files found in:\n"
            f"{folder.resolve()}"
        )

    # --------------------------------------------------------
    # Sort by frame number
    # --------------------------------------------------------

    frames.sort(key=lambda x: x[0])

    print()
    print("=" * 60)
    print("GIF CREATION")
    print("=" * 60)

    print(f"Found {len(frames)} frames")
    print(f"Folder      : {folder.resolve()}")
    print(f"Sequence    : {frames[0][1]}")
    print(f"First frame : {frames[0][2].name}")
    print(f"Last frame  : {frames[-1][2].name}")
    print(f"dt/frame    : {dt_per_frame} s")
    print(f"GIF speed   : {frame_duration} ms/frame")

    # --------------------------------------------------------
    # Load font
    # --------------------------------------------------------

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24
        )
    except Exception:
        font = ImageFont.load_default()

    # --------------------------------------------------------
    # Create GIF frames
    # --------------------------------------------------------

    gif_frames = []

    first_frame = frames[0][0]

    for frame_number, _, image_path in frames:

        img = Image.open(image_path).convert("RGB")

        draw = ImageDraw.Draw(img)

        # Calculate simulation time
        simulation_time = (
            frame_number - first_frame
        ) * dt_per_frame

        # Add timestamp
        draw.text(
            (10, 10),
            f"t = {simulation_time:.2f} s",
            fill="white",
            font=font,
        )

        gif_frames.append(img)

    # --------------------------------------------------------
    # Save GIF
    # --------------------------------------------------------

    output_path = folder / output_name

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=False,
    )

    print()
    print("GIF created:")
    print(output_path.resolve())
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print()
        print(
            "    python animation_gif.py "
            "<folder> [dt_per_frame]"
        )

        print()
        print("Examples:")
        print()

        print(
            "    python animation_gif.py "
            "outputs/simulation1/results"
        )

        print(
            "    python animation_gif.py "
            "outputs/simulation1/results 0.3"
        )

        sys.exit(1)

    folder = sys.argv[1]

    dt = (
        float(sys.argv[2])
        if len(sys.argv) >= 3
        else 0.1
    )

    create_timestamp_gif(
        folder_path=folder,
        dt_per_frame=dt,
    )
