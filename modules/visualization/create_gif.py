import os
import imageio.v2 as imageio


def create_gif_stream(
    frames_dir,
    output_path="outputs/heat.gif",
    fps=10
):
    """
    Streaming GIF creation (memory-efficient).

    Args:
        frames_dir (str): folder with frame_XXXX.png
        output_path (str): output GIF path
        fps (int): frames per second
    """

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f" Frames folder not found: {frames_dir}")

    files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])

    if not files:
        raise ValueError(" No PNG frames found")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"🎬 Streaming GIF creation from {len(files)} frames...")
    print(f"📁 Source: {frames_dir}")
    print(f"💾 Output: {output_path}")

    # duration = seconds per frame
    duration = 1.0 / fps

    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for i, filename in enumerate(files):
            path = os.path.join(frames_dir, filename)

            try:
                image = imageio.imread(path)
                writer.append_data(image)

                if i % 10 == 0:
                    print(f"   ✔ Added frame {i+1}/{len(files)}")

            except Exception as e:
                print(f"⚠️ Skipping frame {filename}: {e}")

    print(f"✅ GIF saved: {output_path}")
