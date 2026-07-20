import os
import trimesh


def scale_stl_microns_to_meters(stl_path):
    """
    Reads an STL assumed to be in microns (μm),
    converts it to meters,
    and saves it as *_scaled.stl.
    """

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(
            f"File not found: {stl_path}"
        )

    mesh = trimesh.load_mesh(stl_path)

    # μm -> m
    scale_factor = 1e-6

    mesh.apply_scale(scale_factor)

    folder = os.path.dirname(stl_path)
    filename = os.path.basename(stl_path)
    name, ext = os.path.splitext(filename)

    output_path = os.path.join(
        folder,
        f"{name}_scaled{ext}"
    )

    mesh.export(output_path)

    print(f"Input STL : {stl_path}")
    print(f"Output STL: {output_path}")
    print(
        f"Scale factor applied: "
        f"{scale_factor} (μm → m)"
    )

    return output_path


if __name__ == "__main__":

    stl_path = input(
        "Enter STL path: "
    ).strip()

    scale_stl_microns_to_meters(stl_path)
