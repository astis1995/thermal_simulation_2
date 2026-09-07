# create_meshes.py

import os
import traceback

from modules.stl_to_mesh.convert import convert_stl_to_xdmf


SCANS_DIR = "scans"


def find_stl_files(base_dir):
    """
    Recursively find all STL files.
    """
    stl_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".stl"):
                stl_files.append(os.path.join(root, file))

    return stl_files


def process_stl(stl_path):
    """
    Generate mesh for a single STL.
    """

    folder = os.path.dirname(stl_path)
    name = os.path.splitext(os.path.basename(stl_path))[0]

    output_prefix = os.path.join(folder, f"{name}_mesh")

    xdmf_path = output_prefix + ".xdmf"

    # Skip if already exists (cache)
    if os.path.exists(xdmf_path):
        print(f"⚡ Skipping (already exists): {xdmf_path}")
        return

    print(f"\n🔧 Processing: {stl_path}")

    try:
        convert_stl_to_xdmf(
            stl_path=stl_path,
            output_path=output_prefix,
            lc_min=0.005,
            lc_max=0.02
        )

        print(f"✅ Success: {xdmf_path}")

    except Exception as e:
        print(f" Failed: {stl_path}")
        print(f"   Error: {e}")

        # Print full traceback for debugging
        traceback.print_exc()

        # Continue execution (important)
        return


def main():
    print("🚀 Batch mesh generation started\n")

    stl_files = find_stl_files(SCANS_DIR)

    print(f"📦 Found {len(stl_files)} STL files\n")

    success = 0
    failed = 0

    for stl in stl_files:
        try:
            process_stl(stl)
            success += 1
        except Exception:
            failed += 1

    print("\n📊 Summary:")
    print(f"   ✅ Processed: {success}")
    print(f"    Failed: {failed}")


if __name__ == "__main__":
    main()
