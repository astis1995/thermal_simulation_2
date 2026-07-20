# modules/stl_to_mesh/generate_mesh_from_stl.py

from .surface import generate_surface_mesh
from .volume import generate_volume_mesh


def generate_mesh_from_stl(
    stl_path: str,
    output_dir: str,
    output_filename: str = "mesh.xdmf",
    lc_min: float = 0.2,
    lc_max: float = 0.5,
    feature_angle: float = 40.0,
    geometry_representation: str = "surface",
    unit: str = "mm",
):
    """
    Generate either a surface mesh or a volumetric tetrahedral mesh
    from an STL file.

    Parameters
    ----------
    geometry_representation
        "surface" : Generate a triangular surface mesh.

        "volume" : Generate a tetrahedral volumetric mesh.
    """

    representation = geometry_representation.lower()

    if representation == "surface":

        return generate_surface_mesh(
            stl_path=stl_path,
            output_dir=output_dir,
            output_filename=output_filename,
            lc_min=lc_min,
            lc_max=lc_max,
            unit=unit,
        )

    elif representation == "volume":

        return generate_volume_mesh(
            stl_path=stl_path,
            output_dir=output_dir,
            output_filename=output_filename,
            lc_min=lc_min,
            lc_max=lc_max,
            feature_angle=feature_angle,
            unit=unit,
        )

    else:

        raise ValueError(
            f"Unknown geometry representation: "
            f"{geometry_representation}\n"
            "Expected 'surface' or 'volume'."
        )
