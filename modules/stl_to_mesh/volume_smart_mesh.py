# modules/stl_to_mesh/volume_smart_mesh.py

import os
import math

import gmsh

from .common import (
    convert_model_to_mesh,
    write_xdmf,
    print_mesh_info,
)


# ============================================================
# UNIT CONVERSION
# ============================================================

UNIT_SCALE = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
}


def _unit_scale(unit):
    """Return conversion factor from STL/Gmsh units to meters."""
    try:
        return UNIT_SCALE[unit.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported unit '{unit}'. "
            f"Supported units: {list(UNIT_SCALE)}"
        )


# ============================================================
# SOURCE PARAMETERS
# ============================================================

def _get_source_parameters(source_config):
    """
    Extract Gaussian source parameters.

    Source coordinates and radius are specified in meters.
    """
    if source_config is None:
        return None

    source_type = str(
        source_config.get("type", "")
    ).lower()

    if source_type != "gaussian":
        print(
            f"\nSource type '{source_type}' is not Gaussian."
        )
        print("Beam refinement disabled.")
        return None

    center = source_config.get(
        "center",
        [0.0, 0.0, 0.0],
    )

    radius = float(
        source_config.get(
            "radius",
            0.002,
        )
    )

    direction = source_config.get(
        "direction",
        [0.0, 0.0, -1.0],
    )

    return {
        "type": source_type,
        "center": [
            float(v) for v in center
        ],
        "radius": radius,
        "direction": [
            float(v) for v in direction
        ],
    }


def _normalize(vector):
    """Normalize a 3-component vector."""
    length = math.sqrt(
        sum(float(v) ** 2 for v in vector)
    )

    if length <= 0.0:
        raise ValueError(
            "Source direction cannot be zero."
        )

    return [
        float(v) / length
        for v in vector
    ]


# ============================================================
# BOUNDING BOX
# ============================================================

def _get_geometry_bbox():
    """
    Get the bounding box of the imported STL.

    The bounding box is used only for diagnostics,
    mesh sizing and source-point clamping.

    It is NEVER converted into computational geometry.
    """
    bbox = gmsh.model.getBoundingBox(
        -1,
        -1,
    )

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise RuntimeError(
            "Invalid STL bounding box."
        )

    return bbox, Lx, Ly, Lz


# ============================================================
# MESH RESOLUTION
# ============================================================

def _calculate_mesh_size(
    Lx,
    Ly,
    Lz,
    lc_min,
    lc_max,
    source,
    scale,
):
    """
    Calculate the target mesh sizes.

    The bounding box is used only to report geometry dimensions.
    It does not impose any topology on the STL.
    """
    lc_min = float(lc_min)
    lc_max = float(lc_max)

    if lc_min <= 0:
        raise ValueError(
            "lc_min must be greater than zero."
        )

    if lc_max < lc_min:
        raise ValueError(
            "lc_max must be greater than or equal to lc_min."
        )

    target_size = lc_max
    fine_size = target_size

    if source is not None:
        radius_gmsh = (
            float(source["radius"]) / float(scale)
        )

        if radius_gmsh <= 0:
            raise ValueError(
                "Source radius must be greater than zero."
            )

        # Approximately six elements across the source radius.
        beam_target = radius_gmsh / 6.0

        fine_size = max(
            lc_min,
            min(
                lc_max,
                beam_target,
            ),
        )

    print("\n=== MESH RESOLUTION ===")
    print(
        f"Bounding-box dimensions : "
        f"{Lx:.6e}, "
        f"{Ly:.6e}, "
        f"{Lz:.6e}"
    )
    print(
        f"Requested minimum size  : "
        f"{lc_min:.6e} {('Gmsh units')}"
    )
    print(
        f"Requested maximum size  : "
        f"{lc_max:.6e} {('Gmsh units')}"
    )
    print(
        f"Bulk mesh size          : "
        f"{target_size:.6e}"
    )
    print(
        f"Fine mesh size          : "
        f"{fine_size:.6e}"
    )

    return target_size, fine_size


# ============================================================
# GLOBAL MESH SIZE
# ============================================================

def _configure_global_mesh_size(
    lc_min,
    lc_max,
):
    """
    Configure global Gmsh mesh-size limits.

    These values control mesh density. They do not define geometry.
    """
    lc_min = float(lc_min)
    lc_max = float(lc_max)

    if lc_min <= 0:
        raise ValueError("lc_min must be > 0.")

    if lc_max < lc_min:
        raise ValueError("lc_max must be >= lc_min.")

    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        lc_min,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        lc_max,
    )

    # Point-based sizing is enabled because the reconstructed
    # STL geometry has geometric vertices.
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints",
        1,
    )

    # Let Gmsh use boundary extension, but do not add an
    # unrelated curvature-based refinement requirement.
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        0,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        1,
    )

    print("\n=== GLOBAL MESH SIZE ===")
    print(
        f"MeshSizeMin : "
        f"{gmsh.option.getNumber('Mesh.MeshSizeMin')}"
    )
    print(
        f"MeshSizeMax : "
        f"{gmsh.option.getNumber('Mesh.MeshSizeMax')}"
    )


def _set_surface_point_sizes(size):
    """
    Explicitly assign the requested target size to reconstructed
    geometric points.

    This is important for STL input: setting only global
    MeshSizeMin/Max does not guarantee that a very coarse
    discrete STL surface will be remeshed at the requested size.
    """
    points = gmsh.model.getEntities(0)

    if not points:
        raise RuntimeError(
            "No geometric points exist after STL reconstruction."
        )

    tags = [tag for dim, tag in points]

    gmsh.model.mesh.setSize(
        [(0, tag) for tag in tags],
        float(size),
    )

    print(
        f"✓ Applied surface point size: {float(size):.6e}"
    )
    print(
        f"  Geometric points sized   : {len(tags):,}"
    )


# ============================================================
# STL SURFACE RECONSTRUCTION / REMESHING
# ============================================================

def _reconstruct_and_remesh_stl(
    feature_angle,
    surface_size,
):
    """
    Convert the imported discrete STL into Gmsh geometric surfaces
    and generate a new 2D surface mesh at the requested resolution.

    IMPORTANT:
        The STL topology is retained.

    We do NOT:
        - create a bounding box,
        - create a rectangular prism,
        - replace the C-channel by a box.

    The workflow is:

        discrete STL
            -> classify surfaces
            -> create geometric parametrization
            -> set mesh size
            -> generate 2D mesh

    The resulting geometric surfaces are then used to build the
    3D volume.
    """
    print("\n======================================")
    print("=== RECONSTRUCTING STL SURFACES ===")
    print("======================================")

    initial_surfaces = gmsh.model.getEntities(2)

    if not initial_surfaces:
        raise RuntimeError(
            "No STL surface entities are available."
        )

    print(
        f"Initial STL surface entities : "
        f"{len(initial_surfaces)}"
    )

    # Gmsh's classifySurfaces operates on the discrete mesh and
    # separates it into smooth/planar surface patches.
    #
    # feature_angle is retained from the original API and now
    # actually controls this classification.
    angle_rad = math.radians(float(feature_angle))

    if angle_rad <= 0.0:
        raise ValueError(
            "feature_angle must be greater than zero."
        )

    if angle_rad >= math.pi:
        raise ValueError(
            "feature_angle must be smaller than 180 degrees."
        )

    print(
        f"Feature angle : {feature_angle:.3f} deg"
    )

    gmsh.model.mesh.classifySurfaces(
        angle_rad,
        True,   # boundary
        True,   # forReparametrization
        math.pi,
    )

    # Build CAD-like parametrizations for the classified discrete
    # surfaces. This is what allows Gmsh to actually remesh the STL
    # surface rather than simply tetrahedralize its original 28 faces.
    try:
        gmsh.model.mesh.createGeometry()
    except Exception as exc:
        raise RuntimeError(
            "Gmsh could not create geometry from the STL surface. "
            "The STL may not be suitable for surface reconstruction."
        ) from exc

    reconstructed_surfaces = gmsh.model.getEntities(2)

    if not reconstructed_surfaces:
        raise RuntimeError(
            "No geometric surfaces were created from the STL."
        )

    print(
        f"Reconstructed surfaces       : "
        f"{len(reconstructed_surfaces)}"
    )

    # Explicitly seed the reconstructed geometry with the requested
    # surface resolution.
    _set_surface_point_sizes(surface_size)

    # Ensure the surface mesher respects the requested limits.
    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        float(surface_size),
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        float(surface_size),
    )

    print("\n=== GENERATING 2D STL SURFACE MESH ===")
    print(
        f"Target surface size : "
        f"{float(surface_size):.6e}"
    )

    try:
        gmsh.model.mesh.generate(2)
    except Exception as exc:
        raise RuntimeError(
            "Gmsh failed to generate the 2D surface mesh."
        ) from exc

    # Diagnostics after surface remeshing.
    node_tags, _, _ = gmsh.model.mesh.getNodes()

    element_types, element_tags, _ = (
        gmsh.model.mesh.getElements(2)
    )

    triangle_count = 0
    total_2d = 0

    for element_type, tags in zip(
        element_types,
        element_tags,
    ):
        total_2d += len(tags)

        # Gmsh type 2 = 3-node linear triangle.
        if element_type == 2:
            triangle_count += len(tags)

    print("\n=== 2D SURFACE MESH RESULT ===")
    print(
        f"Surface nodes    : {len(node_tags):,}"
    )
    print(
        f"2D elements       : {total_2d:,}"
    )
    print(
        f"Linear triangles   : {triangle_count:,}"
    )

    if len(node_tags) < 10:
        raise RuntimeError(
            "Surface remeshing produced suspiciously few nodes. "
            "The STL was not sufficiently refined."
        )

    if triangle_count == 0:
        raise RuntimeError(
            "Surface remeshing produced no linear triangles."
        )

    print(
        "\n✓ STL surface was remeshed before 3D tetrahedralization."
    )

    return reconstructed_surfaces


# ============================================================
# SOURCE REFINEMENT
# ============================================================

def _apply_source_refinement(
    source,
    scale,
    fine_size,
    bbox,
):
    """
    Add a local mesh refinement around the Gaussian source.

    This does not modify the STL geometry.
    """
    if source is None:
        return None

    center = [
        float(v) / float(scale)
        for v in source["center"]
    ]

    radius = (
        float(source["radius"])
        / float(scale)
    )

    fine_size = float(fine_size)

    if radius <= 0:
        raise ValueError(
            "Source radius must be > 0."
        )

    if fine_size <= 0:
        raise ValueError(
            "Fine mesh size must be > 0."
        )

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    point = [
        min(max(center[0], xmin), xmax),
        min(max(center[1], ymin), ymax),
        min(max(center[2], zmin), zmax),
    ]

    fine_radius = max(
        1.5 * radius,
        3.0 * fine_size,
    )

    print("\n======================================")
    print("=== SOURCE REFINEMENT ===")
    print("======================================")

    print(
        f"Source center (Gmsh) : {center}"
    )
    print(
        f"Refinement point     : {point}"
    )
    print(
        f"Fine size            : {fine_size:.6e}"
    )
    print(
        f"Fine radius          : {fine_radius:.6e}"
    )

    # IMPORTANT:
    # The point is created in the geometry kernel only for the
    # mesh-size field. It is NOT used as computational geometry.
    point_tag = gmsh.model.geo.addPoint(
        point[0],
        point[1],
        point[2],
        fine_size,
    )

    gmsh.model.geo.synchronize()

    distance_field = gmsh.model.mesh.field.add(
        "Distance"
    )

    gmsh.model.mesh.field.setNumbers(
        distance_field,
        "NodesList",
        [point_tag],
    )

    threshold_field = gmsh.model.mesh.field.add(
        "Threshold"
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "InField",
        distance_field,
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "SizeMin",
        fine_size,
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "SizeMax",
        fine_size * 5.0,
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "DistMin",
        0.0,
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "DistMax",
        fine_radius,
    )

    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "StopAtDistMax",
        1,
    )

    gmsh.model.mesh.field.setAsBackgroundMesh(
        threshold_field
    )

    print(
        f"✓ Source refinement field: "
        f"{threshold_field}"
    )

    return threshold_field


# ============================================================
# VOLUME FROM RECONSTRUCTED STL
# ============================================================

def _create_volume_from_stl(
    surfaces,
):
    """
    Create a Gmsh volume from the reconstructed STL boundary.

    The reconstructed STL surfaces define the computational
    geometry. No bounding-box reconstruction is performed.
    """
    print("\n======================================")
    print("=== CREATING VOLUME FROM STL ===")
    print("======================================")

    if not surfaces:
        raise RuntimeError(
            "Cannot create volume: no STL surfaces found."
        )

    surface_tags = [
        int(tag)
        for dim, tag in surfaces
        if dim == 2
    ]

    if not surface_tags:
        raise RuntimeError(
            "No 2D STL surfaces found."
        )

    print(
        f"STL surface entities : "
        f"{len(surface_tags)}"
    )
    print(
        f"Surface tags          : "
        f"{surface_tags}"
    )

    # The classified/reconstructed surfaces are now used as the
    # boundary of the volume.
    try:
        surface_loop = (
            gmsh.model.geo.addSurfaceLoop(
                surface_tags
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create a surface loop from the "
            "reconstructed STL. The STL may not form a "
            "valid closed shell."
        ) from exc

    print(
        f"Surface loop          : "
        f"{surface_loop}"
    )

    try:
        volume = gmsh.model.geo.addVolume(
            [surface_loop]
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create a volume from the reconstructed "
            "STL surface loop. Verify that the STL is watertight."
        ) from exc

    print(
        f"Volume                : "
        f"{volume}"
    )

    gmsh.model.geo.synchronize()

    volumes = gmsh.model.getEntities(3)

    if (3, volume) not in volumes:
        raise RuntimeError(
            f"Gmsh volume {volume} was not created."
        )

    boundary = gmsh.model.getBoundary(
        [(3, volume)],
        combined=False,
        oriented=False,
        recursive=False,
    )

    boundary_surfaces = [
        tag
        for dim, tag in boundary
        if dim == 2
    ]

    print(
        f"Volume boundary surfaces : "
        f"{len(boundary_surfaces)}"
    )

    if not boundary_surfaces:
        raise RuntimeError(
            "Created volume has no boundary surfaces."
        )

    print(
        "\n✓ STL geometry preserved."
    )

    return volume


# ============================================================
# PHYSICAL GROUP
# ============================================================

def _create_physical_group(volume):
    """
    Create the physical volume group required by DOLFINx.
    """
    print("\n======================================")
    print("=== PHYSICAL GROUP ===")
    print("======================================")

    existing = gmsh.model.getPhysicalGroups()

    if existing:
        gmsh.model.removePhysicalGroups(
            existing
        )

    physical_tag = gmsh.model.addPhysicalGroup(
        3,
        [volume],
        1,
    )

    gmsh.model.setPhysicalName(
        3,
        physical_tag,
        "Volume",
    )

    groups = gmsh.model.getPhysicalGroups()

    if (3, physical_tag) not in groups:
        raise RuntimeError(
            "Failed to create physical volume group."
        )

    entities = (
        gmsh.model.getEntitiesForPhysicalGroup(
            3,
            physical_tag,
        )
    )

    if volume not in entities:
        raise RuntimeError(
            "Physical group does not contain "
            f"volume {volume}."
        )

    print(
        f"✓ Physical volume group: "
        f"{physical_tag}"
    )

    return physical_tag


# ============================================================
# 3D MESH GENERATION
# ============================================================

def _generate_mesh(
    volume,
    mesh_algorithm=1,
):
    """
    Generate a tetrahedral 3D mesh from the reconstructed
    STL volume.

    The 2D STL surface mesh is generated before this function
    is called.
    """
    print("\n======================================")
    print("=== GENERATING 3D STL MESH ===")
    print("======================================")

    volumes = gmsh.model.getEntities(3)

    if (3, volume) not in volumes:
        raise RuntimeError(
            f"Volume {volume} does not exist."
        )

    gmsh.option.setNumber(
        "Mesh.Algorithm3D",
        int(mesh_algorithm),
    )

    # Keep tetrahedra.
    gmsh.option.setNumber(
        "Mesh.RecombineAll",
        0,
    )

    print(
        "\nCalling Gmsh 3D tetrahedral mesher..."
    )

    gmsh.model.mesh.generate(3)

    node_tags, _, _ = (
        gmsh.model.mesh.getNodes()
    )

    element_types, element_tags, _ = (
        gmsh.model.mesh.getElements(3)
    )

    total_elements = sum(
        len(tags)
        for tags in element_tags
    )

    print("\n=== GMSH MESH RESULT ===")
    print(
        f"Nodes       : "
        f"{len(node_tags):,}"
    )
    print(
        f"3D elements : "
        f"{total_elements:,}"
    )
    print(
        f"Element types : "
        f"{element_types}"
    )

    tetrahedral_count = 0

    for element_type, tags in zip(
        element_types,
        element_tags,
    ):
        properties = (
            gmsh.model.mesh.getElementProperties(
                element_type
            )
        )

        name = properties[0]
        dimension = properties[1]
        order = properties[2]

        print(
            f"  Type {element_type}: "
            f"{name}, "
            f"dim={dimension}, "
            f"order={order}, "
            f"count={len(tags):,}"
        )

        # Gmsh type 4 = 4-node linear tetrahedron.
        if element_type == 4:
            tetrahedral_count += len(tags)

    if len(node_tags) == 0:
        raise RuntimeError(
            "Gmsh generated zero nodes."
        )

    if total_elements == 0:
        raise RuntimeError(
            "Gmsh generated zero 3D elements."
        )

    if tetrahedral_count == 0:
        raise RuntimeError(
            "No linear tetrahedral elements "
            "were generated."
        )

    print(
        f"\n✓ Linear tetrahedra: "
        f"{tetrahedral_count:,}"
    )

    return {
        "nodes": len(node_tags),
        "elements": total_elements,
        "tetrahedra": tetrahedral_count,
        "element_types": list(element_types),
    }


# ============================================================
# MAIN SMART MESHER
# ============================================================

def generate_smart_volume_mesh(
    output_dir,
    output_filename="mesh.xdmf",
    lc_min=0.2,
    lc_max=0.5,
    unit="mm",
    feature_angle=40.0,
    source_config=None,
):
    """
    Generate a tetrahedral volume mesh directly from a
    watertight STL.

    Public signature intentionally preserved for plug-and-play
    compatibility.

    Parameters
    ----------
    output_dir : str
        Directory where the XDMF mesh is written.

    output_filename : str
        XDMF output filename.

    lc_min : float
        Minimum mesh size in STL/Gmsh units.

    lc_max : float
        Maximum mesh size in STL/Gmsh units.

    unit : str
        STL/Gmsh unit: m, cm, mm or um.

    feature_angle : float
        Angle used by Gmsh to classify STL surfaces.

    source_config : dict or None
        Optional Gaussian source configuration.

    IMPORTANT
    ---------
    The imported STL is the computational geometry.

    The bounding box is used only for:
        - diagnostics
        - mesh-size estimation
        - source-point clamping

    It is NEVER converted into a CAD box.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print("\n======================================")
    print("🔷 SMART STL VOLUME MESH")
    print("======================================")

    # ========================================================
    # UNIT SYSTEM
    # ========================================================

    scale = _unit_scale(unit)

    print("\n=== UNIT SYSTEM ===")
    print(
        f"STL/Gmsh unit : {unit}"
    )
    print(
        f"STL → meters  : {scale}"
    )

    # ========================================================
    # IMPORTED STL
    # ========================================================

    surfaces = gmsh.model.getEntities(2)

    if not surfaces:
        raise RuntimeError(
            "No STL surfaces found in Gmsh model."
        )

    print("\n=== IMPORTED STL ===")
    print(
        f"Surface entities : "
        f"{len(surfaces)}"
    )
    print(
        f"Surfaces         : "
        f"{surfaces}"
    )

    # ========================================================
    # GEOMETRY BOUNDING BOX
    # ========================================================

    bbox, Lx, Ly, Lz = (
        _get_geometry_bbox()
    )

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    print("\n=== STL GEOMETRY ===")
    print(
        f"X: {xmin:.6e} -> "
        f"{xmax:.6e}"
    )
    print(
        f"Y: {ymin:.6e} -> "
        f"{ymax:.6e}"
    )
    print(
        f"Z: {zmin:.6e} -> "
        f"{zmax:.6e}"
    )
    print(
        f"Lx = {Lx:.6e} {unit}"
    )
    print(
        f"Ly = {Ly:.6e} {unit}"
    )
    print(
        f"Lz = {Lz:.6e} {unit}"
    )

    # ========================================================
    # SOURCE
    # ========================================================

    source = _get_source_parameters(
        source_config
    )

    if source is not None:
        print("\n=== SOURCE ===")
        print(
            f"Type       : "
            f"{source['type']}"
        )
        print(
            f"Center (m) : "
            f"{source['center']}"
        )
        print(
            f"Radius (m) : "
            f"{source['radius']:.6e}"
        )
        print(
            f"Direction  : "
            f"{source['direction']}"
        )
    else:
        print("\n=== SOURCE ===")
        print(
            "No Gaussian source configured."
        )

    # ========================================================
    # MESH RESOLUTION
    # ========================================================

    bulk_size, fine_size = (
        _calculate_mesh_size(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            lc_min=float(lc_min),
            lc_max=float(lc_max),
            source=source,
            scale=scale,
        )
    )

    _configure_global_mesh_size(
        lc_min=float(lc_min),
        lc_max=bulk_size,
    )

    # ========================================================
    # RECONSTRUCT + REMESH STL SURFACE
    # ========================================================
    #
    # THIS IS THE KEY STEP.
    #
    # The original implementation went directly from a tiny
    # discrete STL surface to gmsh.model.mesh.generate(3).
    #
    # Here we first reconstruct the STL surfaces and generate
    # a proper 2D mesh at the requested resolution.
    #
    # The C-channel topology is therefore retained.
    # ========================================================

    reconstructed_surfaces = (
        _reconstruct_and_remesh_stl(
            feature_angle=float(feature_angle),
            surface_size=bulk_size,
        )
    )

    # ========================================================
    # CREATE 3D VOLUME
    # ========================================================

    print("\n=== BUILDING COMPUTATIONAL VOLUME ===")
    print(
        "Using reconstructed imported STL surface."
    )
    print(
        "Bounding box is NOT used "
        "as computational geometry."
    )

    volume = _create_volume_from_stl(
        reconstructed_surfaces
    )

    # ========================================================
    # SOURCE REFINEMENT
    # ========================================================

    if source is not None:
        _apply_source_refinement(
            source=source,
            scale=scale,
            fine_size=fine_size,
            bbox=bbox,
        )

    # ========================================================
    # PHYSICAL GROUP
    # ========================================================

    _create_physical_group(
        volume
    )

    # ========================================================
    # GENERATE 3D MESH
    # ========================================================

    mesh_info = _generate_mesh(
        volume,
        mesh_algorithm=1,
    )

    # ========================================================
    # CONVERT TO DOLFINX
    # ========================================================

    print(
        "\n🔄 Converting Gmsh → DOLFINx..."
    )

    mesh = convert_model_to_mesh(
        unit
    )

    print(
        "✓ DOLFINx conversion successful"
    )

    # ========================================================
    # MESH DIAGNOSTICS
    # ========================================================

    print_mesh_info(
        mesh
    )

    # ========================================================
    # WRITE XDMF
    # ========================================================

    print("\n💾 Writing XDMF:")
    print(
        xdmf_path
    )

    write_xdmf(
        mesh,
        xdmf_path,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n======================================")
    print("✅ SMART STL VOLUME MESH CREATED")
    print("======================================")

    print(
        f"Nodes       : "
        f"{mesh_info['nodes']:,}"
    )

    print(
        f"Tetrahedra : "
        f"{mesh_info['tetrahedra']:,}"
    )

    print(
        f"Output      : "
        f"{xdmf_path}"
    )

    return xdmf_path
