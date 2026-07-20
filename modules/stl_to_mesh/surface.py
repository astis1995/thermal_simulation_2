import os

import gmsh

from .common import (
    initialize_gmsh,
    finalize_gmsh,
    load_stl,
    convert_model_to_mesh,
    write_xdmf,
    print_mesh_info,
)


def generate_surface_mesh(
    stl_path: str,
    output_dir: str,
    output_filename: str = "mesh.xdmf",
    lc_min: float = 0.2,
    lc_max: float = 0.5,
    unit: str = "mm",
):
    """
    Generate a surface mesh directly from an STL.

    The STL triangles are used directly. No CAD reconstruction
    or volume creation is performed.
    """

    os.makedirs(output_dir, exist_ok=True)

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print(f"📥 STL input : {stl_path}")
    print(f"📤 Mesh output: {xdmf_path}")
    print("🔷 Surface meshing")

    initialize_gmsh(
        lc_min=lc_min,
        lc_max=lc_max,
    )

    try:

        load_stl(stl_path)

        # --------------------------------------------------
        # Generate 2D mesh
        # --------------------------------------------------

        gmsh.model.mesh.generate(2)

        # --------------------------------------------------
        # Physical groups
        # --------------------------------------------------

        surfaces = gmsh.model.getEntities(2)

        if not surfaces:
            raise RuntimeError(
                "No surface entities found."
            )

        for _, tag in surfaces:

            gmsh.model.addPhysicalGroup(
                2,
                [tag],
                tag,
            )

        # --------------------------------------------------
        # Convert to DOLFINx
        # --------------------------------------------------

        mesh = convert_model_to_mesh(unit)

        print_mesh_info(mesh)

        write_xdmf(
            mesh,
            xdmf_path,
        )

        print("✅ Surface mesh created")

        return xdmf_path

    finally:

        finalize_gmsh()
