#!/usr/bin/env python3

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
    Creates an animated GIF from a numbered PNG sequence.

    Accepted filenames:

        metal.0000.png
        temperature.0012.png
        animation1.0045.png

    The GIF is written to the same directory.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder.resolve()}"
        )

    # Allow passing the simulation folder instead of results
    if (folder / "results").is_dir():
        folder = folder / "results"

    # Match: anything.NUMBER.png
    pattern = re.compile(r"^(.+)\.(\d+)\.png$")

    frames = []

    for file in folder.glob("*.png"):

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

    if not frames:
        raise RuntimeError(
            f"No numbered PNG files found in:\n{folder.resolve()}"
        )

    frames.sort(key=lambda x: x[0])

    print(f"\nFound {len(frames)} frames")
    print(f"Folder      : {folder.resolve()}")
    print(f"Sequence    : {frames[0][1]}")
    print(f"First frame : {frames[0][2].name}")
    print(f"Last frame  : {frames[-1][2].name}")

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    gif_frames = []

    first_frame = frames[0][0]

    for frame_number, _, image_path in frames:

        img = Image.open(image_path).convert("RGB")

        draw = ImageDraw.Draw(img)

        simulation_time = (
            frame_number - first_frame
        ) * dt_per_frame

        draw.text(
            (10, 10),
            f"t = {simulation_time:.2f} s",
            fill="white",
            font=font,
        )

        gif_frames.append(img)

    output_path = folder / output_name

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=False,
    )

    print("\nGIF created:")
    print(output_path.resolve())


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("    python animation_gif.py <folder> [dt_per_frame]")
        print()
        print("Examples:")
        print("    python animation_gif.py outputs/simulation1/results")
        print("    python animation_gif.py outputs/simulation1/results/20260629")
        print("    python animation_gif.py outputs/simulation1/results 0.1")
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
