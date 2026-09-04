# modules/stl_to_mesh/volume.py

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

# The current aluminum-bar STL has only 24 unique nodes.
# It is geometrically simple, so it should use the smart
# volume mesher rather than the generic STL pathway.
DEFAULT_SIMPLE_GEOMETRY_MAX_NODES = 100


# ==========================================================
# GEOMETRY COMPLEXITY
# ==========================================================

def get_stl_node_count():
    """
    Return the number of unique mesh nodes in the imported STL.
    """

    node_tags, _, _ = gmsh.model.mesh.getNodes()

    if node_tags is None:
        return 0

    return len(node_tags)


def is_simple_geometry(max_nodes):
    """
    Determine whether the imported STL is considered simple.

    This is only used to choose the meshing strategy.
    It does NOT determine whether the STL is geometrically valid.
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
    Generate a tetrahedral volume mesh from an imported STL.

    Gmsh operates on the STL in its original coordinate units.

    Therefore, if the STL is in mm:

        geometry = mm
        lc_min   = mm
        lc_max   = mm

    Conversion to SI units is performed later by
    convert_model_to_mesh(unit).
    """

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print("🔷 Generic volume meshing")

    # ------------------------------------------------------
    # Mesh size
    # ------------------------------------------------------

    lc_min_gmsh = float(lc_min)
    lc_max_gmsh = float(lc_max)

    if lc_min_gmsh <= 0.0:
        raise ValueError(
            "lc_min must be > 0."
        )

    if lc_max_gmsh < lc_min_gmsh:
        raise ValueError(
            "lc_max must be >= lc_min."
        )

    print("\n=== GMSH MESH SIZE ===")
    print(f"Geometry unit : {unit}")
    print(f"lc_min        : {lc_min_gmsh} {unit}")
    print(f"lc_max        : {lc_max_gmsh} {unit}")
    print("======================\n")

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
    # Configure mesh sizing
    # ------------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        lc_min_gmsh,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        lc_max_gmsh,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints",
        1,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        0,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        1,
    )

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
    # Generate tetrahedral mesh
    # ------------------------------------------------------

    print(
        "🔨 Generating 3D tetrahedral mesh..."
    )

    gmsh.model.mesh.generate(3)

    # ------------------------------------------------------
    # Mesh statistics
    # ------------------------------------------------------

    node_tags, _, _ = (
        gmsh.model.mesh.getNodes()
    )

    element_types, element_tags, _ = (
        gmsh.model.mesh.getElements(3)
    )

    total_3d_elements = sum(
        len(tags)
        for tags in element_tags
    )

    print("\n=== GMSH MESH RESULT ===")

    print(
        f"Nodes        : {len(node_tags)}"
    )

    print(
        f"3D elements  : {total_3d_elements}"
    )

    print(
        f"Element types: {element_types}"
    )

    print("========================\n")

    if total_3d_elements <= 100:
        print("⚠ WARNING:")
        print(
            f"   Only {total_3d_elements} "
            "3D elements were generated."
        )
        print(
            f"   Expected a substantially finer mesh "
            f"for lc_max = {lc_max_gmsh} {unit}."
        )

    # ------------------------------------------------------
    # Physical volume
    # ------------------------------------------------------

    physical_groups = gmsh.model.getPhysicalGroups(
        3
    )

    if not physical_groups:

        gmsh.model.addPhysicalGroup(
            3,
            [volume],
            1,
        )

    # ------------------------------------------------------
    # Convert to DOLFINx
    # ------------------------------------------------------

    mesh = convert_model_to_mesh(
        unit
    )

    print_mesh_info(
        mesh
    )

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

    Simple geometry
        -> volume_smart_mesh.py

    Complex geometry
        -> generic STL volume meshing

    The node-count threshold only determines which meshing
    strategy is used.
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

        load_stl(
            stl_path
        )

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

            result = generate_smart_volume_mesh(
                output_dir=output_dir,
                output_filename=output_filename,
                lc_min=lc_min,
                lc_max=lc_max,
                unit=unit,
                feature_angle=feature_angle,
                source_config=source_config,
            )

            return result

        # ==================================================
        # COMPLEX GEOMETRY
        # ==================================================

        print(
            "\n➡ Using generic STL volume mesher"
        )

        result = generate_generic_volume_mesh(
            output_dir=output_dir,
            output_filename=output_filename,
            lc_min=lc_min,
            lc_max=lc_max,
            unit=unit,
        )

        return result

    finally:

        finalize_gmsh()
