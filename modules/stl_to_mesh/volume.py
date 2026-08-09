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

from .volume_smart_mesh import generate_smart_volume_mesh


# ==========================================================
# DEFAULTS
# ==========================================================

DEFAULT_SIMPLE_GEOMETRY_MAX_NODES = 20


# ==========================================================
# GEOMETRY COMPLEXITY
# ==========================================================

def get_stl_node_count():
    """
    Return the number of unique mesh nodes in the imported STL.

    The STL is loaded into Gmsh as a discrete surface mesh.
    """

    node_tags, _, _ = gmsh.model.mesh.getNodes()

    if node_tags is None:
        return 0

    return len(node_tags)


def is_simple_geometry(max_nodes):
    """
    Determine whether the imported STL is considered simple.

    A geometry is considered simple when the imported STL
    contains <= max_nodes unique mesh nodes.
    """

    n_nodes = get_stl_node_count()

    print("\n=== GEOMETRY COMPLEXITY ===")
    print(f"STL nodes       : {n_nodes}")
    print(f"Simple threshold: {max_nodes}")

    simple = n_nodes <= max_nodes

    if simple:
        print("Geometry type   : SIMPLE")
        print("Meshing method  : SMART")
    else:
        print("Geometry type   : COMPLEX")
        print("Meshing method  : GENERIC")

    print("============================\n")

    return simple


# ==========================================================
# GENERIC VOLUME MESH
# ==========================================================

def generate_generic_volume_mesh(
    output_dir,
    output_filename,
    lc_min,
    lc_max,
    unit,
):
    """
    Existing generic STL -> volume pathway.

    IMPORTANT:
    This intentionally does NOT use:

        classifySurfaces()
        createGeometry()

    because those operations caused problems with
    complicated STL geometries.
    """

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print("🔷 Generic volume meshing")

    # ------------------------------------------------------
    # Imported STL surfaces
    # ------------------------------------------------------

    surfaces = gmsh.model.getEntities(2)

    print("\nImported surfaces:")
    print(surfaces)

    if not surfaces:
        raise RuntimeError(
            "No surface entities found in STL."
        )

    surface_tags = [
        tag
        for _, tag in surfaces
    ]

    print(
        f"Number of surfaces: {len(surface_tags)}"
    )

    print(
        f"Surface tags: {surface_tags}"
    )

    # ------------------------------------------------------
    # Create closed volume
    # ------------------------------------------------------

    surface_loop = gmsh.model.geo.addSurfaceLoop(
        surface_tags
    )

    print(
        f"Surface loop id: {surface_loop}"
    )

    volume = gmsh.model.geo.addVolume(
        [surface_loop]
    )

    print(
        f"Volume id: {volume}"
    )

    gmsh.model.geo.synchronize()

    # ------------------------------------------------------
    # Mesh-size diagnostics
    # ------------------------------------------------------

    print("\n=== GMSH MESH SIZE SETTINGS ===")

    print(
        "MeshSizeMin =",
        gmsh.option.getNumber(
            "Mesh.MeshSizeMin"
        ),
    )

    print(
        "MeshSizeMax =",
        gmsh.option.getNumber(
            "Mesh.MeshSizeMax"
        ),
    )

    print(
        "MeshSizeFromPoints =",
        gmsh.option.getNumber(
            "Mesh.MeshSizeFromPoints"
        ),
    )

    print(
        "MeshSizeFromCurvature =",
        gmsh.option.getNumber(
            "Mesh.MeshSizeFromCurvature"
        ),
    )

    print(
        "MeshSizeExtendFromBoundary =",
        gmsh.option.getNumber(
            "Mesh.MeshSizeExtendFromBoundary"
        ),
    )

    print(
        "================================\n"
    )

    # ------------------------------------------------------
    # Generate tetrahedral mesh
    # ------------------------------------------------------

    gmsh.model.mesh.generate(3)

    # ------------------------------------------------------
    # Physical volume
    # ------------------------------------------------------

    gmsh.model.addPhysicalGroup(
        3,
        [volume],
        1,
    )

    # ------------------------------------------------------
    # Convert to DOLFINx
    # ------------------------------------------------------

    mesh = convert_model_to_mesh(unit)

    print_mesh_info(mesh)

    write_xdmf(
        mesh,
        xdmf_path,
    )

    print(
        "✅ Generic volume mesh created"
    )

    return xdmf_path


# ==========================================================
# MAIN VOLUME MESH FUNCTION
# ==========================================================

def generate_volume_mesh(
    stl_path: str,
    output_dir: str,
    output_filename: str = "mesh.xdmf",
    lc_min: float = 0.2,
    lc_max: float = 0.5,
    feature_angle: float = 40.0,
    unit: str = "mm",
    simple_geometry_max_nodes: int = (
        DEFAULT_SIMPLE_GEOMETRY_MAX_NODES
    ),
    source_config=None,
):
    """
    Generate a volumetric tetrahedral mesh from an STL.

    The STL is automatically classified according to its
    number of mesh nodes.

    Simple geometry
        -> volume_smart_mesh.py

    Complex geometry
        -> generic STL volume meshing

    Parameters
    ----------
    stl_path:
        Path to the STL file.

    output_dir:
        Directory where the XDMF mesh is written.

    output_filename:
        Name of the XDMF output.

    lc_min:
        Minimum characteristic mesh size for the generic
        pathway.

    lc_max:
        Maximum characteristic mesh size for the generic
        pathway.

    feature_angle:
        Retained for compatibility with the existing API.

    unit:
        Units of the original STL.

    simple_geometry_max_nodes:
        Maximum number of STL nodes for the geometry to
        be treated as simple.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print(
        f"📥 STL input : {stl_path}"
    )

    print(
        f"📤 Mesh output: {xdmf_path}"
    )

    print(
        "🔷 Volume meshing"
    )

    # ======================================================
    # INITIALIZE GMSH
    # ======================================================

    initialize_gmsh(
        lc_min=lc_min,
        lc_max=lc_max,
    )

    try:

        # ==================================================
        # LOAD STL
        # ==================================================

        load_stl(stl_path)

        # ==================================================
        # DETECT GEOMETRY
        # ==================================================

        simple = is_simple_geometry(
            max_nodes=simple_geometry_max_nodes
        )

        # ==================================================
        # SIMPLE GEOMETRY
        # ==================================================

        if simple:

            print(
                "\n➡ Using volume_smart_mesh.py"
            )

            return generate_smart_volume_mesh(
                output_dir=output_dir,
                output_filename=output_filename,
                lc_min=lc_min,
                lc_max=lc_max,
                unit=unit,
                feature_angle=feature_angle,
                source_config=source_config,
            )

        # ==================================================
        # COMPLEX GEOMETRY
        # ==================================================

        else:

            print(
                "\n➡ Using generic STL volume mesher"
            )

            return generate_generic_volume_mesh(
                output_dir=output_dir,
                output_filename=output_filename,
                lc_min=lc_min,
                lc_max=lc_max,
                unit=unit,
            )

    finally:

        finalize_gmsh()
