import math
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


def generate_volume_mesh(
    stl_path: str,
    output_dir: str,
    output_filename: str = "mesh.xdmf",
    lc_min: float = 0.2,
    lc_max: float = 0.5,
    feature_angle: float = 40.0,
    unit: str = "mm",
):
    """
    Generate a volumetric tetrahedral mesh from a closed STL.
    """

    os.makedirs(output_dir, exist_ok=True)

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print(f"📥 STL input : {stl_path}")
    print(f"📤 Mesh output: {xdmf_path}")
    print("🔷 Volume meshing")

    initialize_gmsh(
        lc_min=lc_min,
        lc_max=lc_max,
    )

    try:

        load_stl(stl_path)

        print("\nImported entities:")
        print(gmsh.model.getEntities())

        print("\nSurfaces:")
        print(gmsh.model.getEntities(2))
        # --------------------------------------------------
        # Reconstruct CAD geometry
        # --------------------------------------------------

        #angle_rad = math.radians(feature_angle)

        #gmsh.model.mesh.classifySurfaces(  angle_rad,boundary=True, forReparametrization=True,curveAngle=math.pi,)

        #gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()

        print("\nReconstructed surfaces:")
        print(gmsh.model.getEntities(2))

        # --------------------------------------------------
        # Create a closed volume
        # --------------------------------------------------

        surfaces = gmsh.model.getEntities(2)
        print(f"Number of reconstructed surfaces: {len(surfaces)}")
        print("Surface tags:", [tag for _, tag in surfaces])


        if not surfaces:
            raise RuntimeError(
                "No reconstructed surfaces found."
            )

        surface_tags = [
            tag for _, tag in surfaces
        ]

        surface_loop = gmsh.model.geo.addSurfaceLoop(
            surface_tags
        )
        print(f"Surface loop id: {surface_loop}")
        volume = gmsh.model.geo.addVolume(
            [surface_loop]
        )
        print(f"Volume id: {volume}")
        gmsh.model.geo.synchronize()

        # --------------------------------------------------
        # Generate tetrahedral mesh
        # --------------------------------------------------

        gmsh.model.mesh.generate(3)

        # --------------------------------------------------
        # Physical groups
        # --------------------------------------------------

        gmsh.model.addPhysicalGroup(
            3,
            [volume],
            1,
        )

        mesh = convert_model_to_mesh(unit)

        print_mesh_info(mesh)

        write_xdmf(
            mesh,
            xdmf_path,
        )

        print("✅ Volume mesh created")

        return xdmf_path

    finally:

        finalize_gmsh()
