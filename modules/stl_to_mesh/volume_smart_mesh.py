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
    """Return conversion factor from the specified unit to meters."""

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
        print(
            "Beam refinement disabled."
        )
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

    if length == 0.0:
        raise ValueError(
            "Source direction cannot be zero."
        )

    return [
        float(v) / length
        for v in vector
    ]


# ============================================================
# ILLUMINATED FACE
# ============================================================

def _find_illuminated_face(
    bbox,
    center_gmsh,
    direction,
):
    """
    Determine which bounding-box face receives the beam.

    direction = direction of propagation.

    Therefore:

        -Z propagation -> +Z illuminated
        +Z propagation -> -Z illuminated

        -X propagation -> +X illuminated
        +X propagation -> -X illuminated

        -Y propagation -> +Y illuminated
        +Y propagation -> -Y illuminated
    """

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    direction = _normalize(direction)

    dx, dy, dz = direction

    cx, cy, cz = center_gmsh

    components = {
        "X": abs(dx),
        "Y": abs(dy),
        "Z": abs(dz),
    }

    axis = max(
        components,
        key=components.get,
    )

    if axis == "X":

        if dx < 0:
            face = "+X"
            point = [
                xmax,
                min(max(cy, ymin), ymax),
                min(max(cz, zmin), zmax),
            ]
        else:
            face = "-X"
            point = [
                xmin,
                min(max(cy, ymin), ymax),
                min(max(cz, zmin), zmax),
            ]

    elif axis == "Y":

        if dy < 0:
            face = "+Y"
            point = [
                min(max(cx, xmin), xmax),
                ymax,
                min(max(cz, zmin), zmax),
            ]
        else:
            face = "-Y"
            point = [
                min(max(cx, xmin), xmax),
                ymin,
                min(max(cz, zmin), zmax),
            ]

    else:

        if dz < 0:
            face = "+Z"
            point = [
                min(max(cx, xmin), xmax),
                min(max(cy, ymin), ymax),
                zmax,
            ]
        else:
            face = "-Z"
            point = [
                min(max(cx, xmin), xmax),
                min(max(cy, ymin), ymax),
                zmin,
            ]

    print("\n=== ILLUMINATED SURFACE ===")

    print(
        f"Propagation direction : "
        f"[{dx:.6g}, {dy:.6g}, {dz:.6g}]"
    )

    print(
        f"Illuminated face       : {face}"
    )

    print(
        f"Beam center (Gmsh)     : "
        f"{center_gmsh}"
    )

    print(
        f"Surface point          : "
        f"{point}"
    )

    return face, point


# ============================================================
# MESH RESOLUTION
# ============================================================

def _calculate_resolution(
    Lx,
    Ly,
    Lz,
    lc_min,
    lc_max,
    source,
    scale,
):
    """
    Determine independent X/Y/Z mesh resolutions.

    Lx, Ly, Lz and lc_min/lc_max are in Gmsh/STL units.

    This function intentionally does NOT assume isotropic
    elements.
    """

    # --------------------------------------------------------
    # Default bulk spacing
    # --------------------------------------------------------

    bulk_xy = float(lc_max)

    # --------------------------------------------------------
    # Thickness resolution
    # --------------------------------------------------------

    # Use the bulk mesh size through the thickness,
    # but guarantee a minimum number of layers.

    MIN_Z_LAYERS = 8

    nz = max(
        MIN_Z_LAYERS,
        int(math.ceil(Lz / lc_max)),
    )

    dz = Lz / nz

    # Never allow the thickness spacing to exceed lc_min
    # if the body is thin.
    if dz > float(lc_min):
        nz = max(
            MIN_Z_LAYERS,
            int(math.ceil(Lz / float(lc_min))),
        )

        dz = Lz / nz

    # --------------------------------------------------------
    # Default X/Y resolution
    # --------------------------------------------------------

    nx_bulk = max(
        2,
        int(math.ceil(Lx / bulk_xy)),
    )

    ny_bulk = max(
        2,
        int(math.ceil(Ly / bulk_xy)),
    )

    # --------------------------------------------------------
    # Beam refinement
    # --------------------------------------------------------

    if source is None:

        nx = nx_bulk
        ny = ny_bulk

        fine_xy = bulk_xy

        beam_width = None

    else:

        radius_gmsh = (
            source["radius"] / scale
        )

        # ----------------------------------------------------
        # Target approximately 6 elements across
        # the beam radius.
        #
        # This is deliberately conservative.
        # ----------------------------------------------------

        beam_target = radius_gmsh / 6.0

        fine_xy = max(
            float(lc_min),
            min(
                float(lc_max),
                beam_target,
            ),
        )

        # Number of cells required to resolve the
        # complete X/Y dimensions at fine resolution.
        #
        # These are NOT necessarily all generated as
        # fine cells. They are used to determine the
        # target resolution around the beam.
        nx_fine = max(
            2,
            int(math.ceil(Lx / fine_xy)),
        )

        ny_fine = max(
            2,
            int(math.ceil(Ly / fine_xy)),
        )

        # For now, use the conservative resolution.
        #
        # The actual beam-local partition is created
        # below using transfinite blocks.
        nx = nx_bulk
        ny = ny_bulk

        beam_width = 2.0 * radius_gmsh

    return {
        "nx_bulk": nx_bulk,
        "ny_bulk": ny_bulk,
        "nz": nz,
        "dz": dz,
        "fine_xy": fine_xy,
        "beam_width": beam_width,
    }


# ============================================================
# SIMPLE STRUCTURED BLOCK MESH
# ============================================================

def _create_structured_box(
    xmin,
    ymin,
    zmin,
    xmax,
    ymax,
    zmax,
):
    """
    Create an explicit CAD box using Gmsh's GEO kernel.

    The box is constructed from:
        8 points
        12 lines
        6 plane surfaces
        1 closed volume

    This is intentionally explicit so that the resulting
    geometry has a proper CAD topology suitable for
    transfinite meshing.
    """

    print("\n======================================")
    print("=== CREATING STRUCTURED CAD BOX ===")
    print("======================================")

    # --------------------------------------------------
    # 1. Eight corner points
    # --------------------------------------------------

    p000 = gmsh.model.geo.addPoint(
        xmin, ymin, zmin
    )

    p100 = gmsh.model.geo.addPoint(
        xmax, ymin, zmin
    )

    p110 = gmsh.model.geo.addPoint(
        xmax, ymax, zmin
    )

    p010 = gmsh.model.geo.addPoint(
        xmin, ymax, zmin
    )

    p001 = gmsh.model.geo.addPoint(
        xmin, ymin, zmax
    )

    p101 = gmsh.model.geo.addPoint(
        xmax, ymin, zmax
    )

    p111 = gmsh.model.geo.addPoint(
        xmax, ymax, zmax
    )

    p011 = gmsh.model.geo.addPoint(
        xmin, ymax, zmax
    )

    points = [
        p000,
        p100,
        p110,
        p010,
        p001,
        p101,
        p111,
        p011,
    ]

    print(
        f"✓ Created {len(points)} corner points"
    )

    # --------------------------------------------------
    # 2. Twelve edges
    # --------------------------------------------------

    # Bottom
    l01 = gmsh.model.geo.addLine(
        p000, p100
    )

    l12 = gmsh.model.geo.addLine(
        p100, p110
    )

    l23 = gmsh.model.geo.addLine(
        p110, p010
    )

    l30 = gmsh.model.geo.addLine(
        p010, p000
    )

    # Top
    l45 = gmsh.model.geo.addLine(
        p001, p101
    )

    l56 = gmsh.model.geo.addLine(
        p101, p111
    )

    l67 = gmsh.model.geo.addLine(
        p111, p011
    )

    l74 = gmsh.model.geo.addLine(
        p011, p001
    )

    # Vertical
    l04 = gmsh.model.geo.addLine(
        p000, p001
    )

    l15 = gmsh.model.geo.addLine(
        p100, p101
    )

    l26 = gmsh.model.geo.addLine(
        p110, p111
    )

    l37 = gmsh.model.geo.addLine(
        p010, p011
    )

    lines = [
        l01, l12, l23, l30,
        l45, l56, l67, l74,
        l04, l15, l26, l37,
    ]

    print(
        f"✓ Created {len(lines)} boundary curves"
    )

    # --------------------------------------------------
    # 3. Six plane surfaces
    # --------------------------------------------------

    # Bottom: z = zmin
    loop_bottom = gmsh.model.geo.addCurveLoop(
        [l01, l12, l23, l30]
    )

    surface_bottom = gmsh.model.geo.addPlaneSurface(
        [loop_bottom]
    )

    # Top: z = zmax
    loop_top = gmsh.model.geo.addCurveLoop(
        [l45, l56, l67, l74]
    )

    surface_top = gmsh.model.geo.addPlaneSurface(
        [loop_top]
    )

    # Front: y = ymin
    loop_front = gmsh.model.geo.addCurveLoop(
        [l01, l15, -l45, -l04]
    )

    surface_front = gmsh.model.geo.addPlaneSurface(
        [loop_front]
    )

    # Back: y = ymax
    loop_back = gmsh.model.geo.addCurveLoop(
        [l23, l37, -l67, -l26]
    )

    surface_back = gmsh.model.geo.addPlaneSurface(
        [loop_back]
    )

    # Left: x = xmin
    loop_left = gmsh.model.geo.addCurveLoop(
        [l30, l04, -l74, -l37]
    )

    surface_left = gmsh.model.geo.addPlaneSurface(
        [loop_left]
    )

    # Right: x = xmax
    loop_right = gmsh.model.geo.addCurveLoop(
        [l12, l26, -l56, -l15]
    )

    surface_right = gmsh.model.geo.addPlaneSurface(
        [loop_right]
    )

    surfaces = [
        surface_bottom,
        surface_top,
        surface_front,
        surface_back,
        surface_left,
        surface_right,
    ]

    print(
        f"✓ Created {len(surfaces)} plane surfaces"
    )

    # --------------------------------------------------
    # 4. Closed surface loop
    # --------------------------------------------------

    surface_loop = gmsh.model.geo.addSurfaceLoop(
        surfaces
    )

    print(
        f"✓ Surface loop: {surface_loop}"
    )

    # --------------------------------------------------
    # 5. Volume
    # --------------------------------------------------

    volume = gmsh.model.geo.addVolume(
        [surface_loop]
    )

    print(
        f"✓ Volume: {volume}"
    )

    # --------------------------------------------------
    # 6. Synchronize CAD kernel
    # --------------------------------------------------

    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # 7. Verify topology
    # --------------------------------------------------

    created_points = gmsh.model.getEntities(0)
    created_curves = gmsh.model.getEntities(1)
    created_surfaces = gmsh.model.getEntities(2)
    created_volumes = gmsh.model.getEntities(3)

    print("\n=== CAD TOPOLOGY ===")
    print(
        f"Points    : {len(created_points)}"
    )
    print(
        f"Curves    : {len(created_curves)}"
    )
    print(
        f"Surfaces  : {len(created_surfaces)}"
    )
    print(
        f"Volumes   : {len(created_volumes)}"
    )

    if (3, volume) not in created_volumes:
        raise RuntimeError(
            "Structured box volume was not created."
        )

    # Verify that the volume really has six surfaces.
    volume_boundary = gmsh.model.getBoundary(
        [(3, volume)],
        combined=False,
        oriented=False,
        recursive=False,
    )

    volume_surfaces = [
        tag
        for dim, tag in volume_boundary
        if dim == 2
    ]

    print(
        f"Volume boundary surfaces: "
        f"{len(volume_surfaces)}"
    )

    if len(volume_surfaces) != 6:
        raise RuntimeError(
            "Structured box volume should have "
            f"6 boundary surfaces, found "
            f"{len(volume_surfaces)}."
        )

    print(
        "\n✓ Structured CAD box created successfully"
    )

    return volume

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
    Generate a volumetric mesh for a simple geometry.

    IMPORTANT:

    This function is intended only for geometries classified
    as SIMPLE by volume.py.

    It does not call createGeometry() on the imported STL.
    """

    del feature_angle

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print(
        "\n======================================"
    )

    print(
        "🔷 SMART VOLUME MESH"
    )

    print(
        "======================================"
    )

    # ========================================================
    # UNIT SYSTEM
    # ========================================================

    scale = _unit_scale(unit)

    print(
        "\n=== UNIT SYSTEM ==="
    )

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
            "Smart volume mesh: no STL surfaces found."
        )

    print(
        "\n=== IMPORTED STL ==="
    )

    print(
        f"Surfaces: {surfaces}"
    )

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    bbox = gmsh.model.getBoundingBox(
        -1,
        -1,
    )

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

    print(
        "\n=== SIMPLE GEOMETRY ==="
    )

    print(
        f"X: {xmin:.6e} -> {xmax:.6e}"
    )

    print(
        f"Y: {ymin:.6e} -> {ymax:.6e}"
    )

    print(
        f"Z: {zmin:.6e} -> {zmax:.6e}"
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

    print(
        "\n=== SOURCE REFINEMENT ==="
    )

    if source is None:

        print(
            "No Gaussian source."
        )

        center_gmsh = None
        illuminated_face = None
        surface_point = None

    else:

        center_m = source["center"]

        center_gmsh = [
            v / scale
            for v in center_m
        ]

        (
            illuminated_face,
            surface_point,
        ) = _find_illuminated_face(
            bbox,
            center_gmsh,
            source["direction"],
        )

        radius_gmsh = (
            source["radius"] / scale
        )

        print(
            f"Source type          : "
            f"{source['type']}"
        )

        print(
            f"Source radius (m)    : "
            f"{source['radius']:.6e}"
        )

        print(
            f"Source radius (Gmsh): "
            f"{radius_gmsh:.6e}"
        )

        print(
            f"Direction            : "
            f"{source['direction']}"
        )

        print(
            f"Illuminated face     : "
            f"{illuminated_face}"
        )

    # ========================================================
    # RESOLUTION
    # ========================================================

    resolution = _calculate_resolution(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        lc_min=float(lc_min),
        lc_max=float(lc_max),
        source=source,
        scale=scale,
    )

    nx_bulk = resolution["nx_bulk"]
    ny_bulk = resolution["ny_bulk"]
    nz = resolution["nz"]

    fine_xy = resolution["fine_xy"]
    dz = resolution["dz"]

    print(
        "\n=== SMART MESH RESOLUTION ==="
    )

    print(
        f"Bulk X spacing     : "
        f"{Lx / nx_bulk:.6e} {unit}"
    )

    print(
        f"Bulk Y spacing     : "
        f"{Ly / ny_bulk:.6e} {unit}"
    )

    print(
        f"Z spacing          : "
        f"{dz:.6e} {unit}"
    )

    print(
        f"Fine XY target     : "
        f"{fine_xy:.6e} {unit}"
    )

    print(
        f"NX                 : "
        f"{nx_bulk}"
    )

    print(
        f"NY                 : "
        f"{ny_bulk}"
    )

    print(
        f"NZ                 : "
        f"{nz}"
    )

    print(
        f"Estimated cells    : "
        f"{nx_bulk * ny_bulk * nz}"
    )

    # ========================================================
    # REMOVE IMPORTED STL SURFACE FROM ACTIVE MODEL
    #
    # The simple geometry is rebuilt from the exact STL bbox.
    #
    # This avoids the problematic STL parametrization path.
    # ========================================================

    print(
        "\n=== BUILDING SIMPLE VOLUME ==="
    )

    print(
        "Using bounding-box geometry."
    )

    print(
        "Imported STL is used only to determine dimensions."
    )

    # ========================================================
    # CREATE STRUCTURED BOX
    # ========================================================

    volume = _create_structured_box(
        xmin=xmin,
        ymin=ymin,
        zmin=zmin,
        xmax=xmax,
        ymax=ymax,
        zmax=zmax,
        nx=nx_bulk,
        ny=ny_bulk,
        nz=nz,
    )

    # ========================================================
    # PHYSICAL GROUP
    # ========================================================

    print(
        "\n=== PHYSICAL GROUPS ==="
    )

    gmsh.model.addPhysicalGroup(
        3,
        [volume],
        1,
    )

    print(
        "✓ Physical volume group: 1"
    )

    # ========================================================
    # MESH
    # ========================================================

    print(
        "\n🔷 Generating structured tetrahedral mesh..."
    )

    # Transfinite volumes are naturally generated as
    # structured elements. We deliberately do NOT recombine
    # them into hexahedra because the DOLFINx solver expects
    # tetrahedral volume cells in the current pipeline.

    gmsh.model.mesh.generate(
        3
    )

    # ========================================================
    # GMSH DIAGNOSTICS
    # ========================================================

    node_tags, _, _ = (
        gmsh.model.mesh.getNodes()
    )

    element_types, element_tags, _ = (
        gmsh.model.mesh.getElements(
            3
        )
    )

    total_volume_elements = sum(
        len(tags)
        for tags in element_tags
    )

    print(
        "\n=== GMSH MESH RESULT ==="
    )

    print(
        f"Gmsh nodes       : "
        f"{len(node_tags)}"
    )

    print(
        f"3D elements      : "
        f"{total_volume_elements}"
    )

    print(
        f"Element types    : "
        f"{element_types}"
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

    print(
        "\n💾 Writing XDMF:"
    )

    print(
        xdmf_path
    )

    write_xdmf(
        mesh,
        xdmf_path,
    )

    print(
        "\n======================================"
    )

    print(
        "✅ SMART VOLUME MESH CREATED"
    )

    print(
        "======================================"
    )

    return xdmf_path
def _calculate_structured_divisions(
    lx,
    ly,
    lz,
    target_size,
    min_divisions=2,
    max_divisions=500,
):
    """
    Calculate structured mesh divisions from physical dimensions.

    The target element size is applied independently to each
    coordinate direction:

        nx ≈ Lx / target_size
        ny ≈ Ly / target_size
        nz ≈ Lz / target_size

    This keeps the mesh approximately isotropic.

    Parameters
    ----------
    lx, ly, lz : float
        Geometry dimensions in Gmsh units.

    target_size : float
        Desired approximate element size in the same units
        as lx, ly and lz.

    min_divisions : int
        Minimum number of divisions per direction.

    max_divisions : int
        Safety limit to prevent accidental enormous meshes.

    Returns
    -------
    nx, ny, nz : tuple[int, int, int]
        Number of divisions along X, Y and Z.
    """

    print("\n=== STRUCTURED MESH RESOLUTION ===")

    if target_size <= 0:
        raise ValueError(
            "target_size must be greater than zero."
        )

    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(
            "Geometry dimensions must be positive."
        )

    # --------------------------------------------------
    # Calculate divisions from physical dimensions
    # --------------------------------------------------

    nx = max(
        min_divisions,
        int(round(lx / target_size)),
    )

    ny = max(
        min_divisions,
        int(round(ly / target_size)),
    )

    nz = max(
        min_divisions,
        int(round(lz / target_size)),
    )

    # --------------------------------------------------
    # Safety limit
    # --------------------------------------------------

    nx = min(nx, max_divisions)
    ny = min(ny, max_divisions)
    nz = min(nz, max_divisions)

    # --------------------------------------------------
    # Actual resulting spacing
    # --------------------------------------------------

    hx = lx / nx
    hy = ly / ny
    hz = lz / nz

    print(
        f"Geometry dimensions : "
        f"{lx:.6e}, {ly:.6e}, {lz:.6e}"
    )

    print(
        f"Requested size      : "
        f"{target_size:.6e}"
    )

    print("\nDivisions:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")

    print("\nActual element spacing:")
    print(f"  hx = {hx:.6e}")
    print(f"  hy = {hy:.6e}")
    print(f"  hz = {hz:.6e}")

    print("\nEstimated structured nodes:")
    print(
        f"  {(nx + 1) * (ny + 1) * (nz + 1):,}"
    )

    print("\nEstimated tetrahedra:")
    print(
        f"  ~{6 * nx * ny * nz:,}"
    )

    return nx, ny, nz
def _calculate_structured_divisions(
    lx,
    ly,
    lz,
    target_size,
    min_divisions=2,
    max_divisions=500,
):
    """
    Calculate structured mesh divisions from the physical
    dimensions of the geometry.

    The requested target element size is applied independently
    to X, Y and Z, producing approximately isotropic elements.

    Parameters
    ----------
    lx, ly, lz : float
        Geometry dimensions in Gmsh units.

    target_size : float
        Desired element size in the same units as lx, ly and lz.

    min_divisions : int
        Minimum number of divisions in each direction.

    max_divisions : int
        Maximum number of divisions in each direction.
        This prevents accidentally generating enormous meshes.

    Returns
    -------
    nx, ny, nz : int
        Number of divisions along X, Y and Z.
    """

    print("\n=== STRUCTURED MESH RESOLUTION ===")

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if target_size <= 0:
        raise ValueError(
            "target_size must be greater than zero."
        )

    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(
            "Geometry dimensions must be positive."
        )

    if min_divisions < 1:
        raise ValueError(
            "min_divisions must be at least 1."
        )

    if max_divisions < min_divisions:
        raise ValueError(
            "max_divisions must be >= min_divisions."
        )

    # --------------------------------------------------
    # Calculate divisions
    # --------------------------------------------------

    nx = int(round(lx / target_size))
    ny = int(round(ly / target_size))
    nz = int(round(lz / target_size))

    # --------------------------------------------------
    # Enforce minimum
    # --------------------------------------------------

    nx = max(nx, min_divisions)
    ny = max(ny, min_divisions)
    nz = max(nz, min_divisions)

    # --------------------------------------------------
    # Enforce maximum
    # --------------------------------------------------

    nx = min(nx, max_divisions)
    ny = min(ny, max_divisions)
    nz = min(nz, max_divisions)

    # --------------------------------------------------
    # Actual mesh spacing
    # --------------------------------------------------

    hx = lx / nx
    hy = ly / ny
    hz = lz / nz

    # --------------------------------------------------
    # Estimated mesh size
    # --------------------------------------------------

    estimated_nodes = (
        (nx + 1)
        * (ny + 1)
        * (nz + 1)
    )

    estimated_tets = (
        6
        * nx
        * ny
        * nz
    )

    # --------------------------------------------------
    # Feedback
    # --------------------------------------------------

    print(
        f"Geometry:"
        f" Lx={lx:.6e},"
        f" Ly={ly:.6e},"
        f" Lz={lz:.6e}"
    )

    print(
        f"Requested target size : "
        f"{target_size:.6e}"
    )

    print("\nDivisions:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")

    print("\nActual spacing:")
    print(f"  hx = {hx:.6e}")
    print(f"  hy = {hy:.6e}")
    print(f"  hz = {hz:.6e}")

    print("\nEstimated mesh:")
    print(
        f"  Nodes     : "
        f"{estimated_nodes:,}"
    )

    print(
        f"  Tetrahedra: "
        f"~{estimated_tets:,}"
    )

    # --------------------------------------------------
    # Check isotropy
    # --------------------------------------------------

    h_min = min(hx, hy, hz)
    h_max = max(hx, hy, hz)

    if h_min > 0:
        anisotropy_ratio = h_max / h_min
    else:
        anisotropy_ratio = float("inf")

    print(
        f"\nSpacing ratio h_max/h_min : "
        f"{anisotropy_ratio:.3f}"
    )

    if anisotropy_ratio > 1.25:
        print(
            "⚠ Mesh spacing is not perfectly isotropic."
        )
    else:
        print(
            "✓ Mesh spacing is approximately isotropic."
        )

    print(
        "======================================"
    )

    return nx, ny, nz
def _calculate_structured_divisions(
    lx,
    ly,
    lz,
    target_size,
    min_divisions=2,
    max_divisions=500,
):
    """
    Calculate structured mesh divisions from the physical
    dimensions of the geometry.

    The requested target element size is applied independently
    to X, Y and Z, producing approximately isotropic elements.

    Parameters
    ----------
    lx, ly, lz : float
        Geometry dimensions in Gmsh units.

    target_size : float
        Desired element size in the same units as lx, ly and lz.

    min_divisions : int
        Minimum number of divisions in each direction.

    max_divisions : int
        Maximum number of divisions in each direction.
        This prevents accidentally generating enormous meshes.

    Returns
    -------
    nx, ny, nz : int
        Number of divisions along X, Y and Z.
    """

    print("\n=== STRUCTURED MESH RESOLUTION ===")

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if target_size <= 0:
        raise ValueError(
            "target_size must be greater than zero."
        )

    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(
            "Geometry dimensions must be positive."
        )

    if min_divisions < 1:
        raise ValueError(
            "min_divisions must be at least 1."
        )

    if max_divisions < min_divisions:
        raise ValueError(
            "max_divisions must be >= min_divisions."
        )

    # --------------------------------------------------
    # Calculate divisions
    # --------------------------------------------------

    nx = int(round(lx / target_size))
    ny = int(round(ly / target_size))
    nz = int(round(lz / target_size))

    # --------------------------------------------------
    # Enforce minimum
    # --------------------------------------------------

    nx = max(nx, min_divisions)
    ny = max(ny, min_divisions)
    nz = max(nz, min_divisions)

    # --------------------------------------------------
    # Enforce maximum
    # --------------------------------------------------

    nx = min(nx, max_divisions)
    ny = min(ny, max_divisions)
    nz = min(nz, max_divisions)

    # --------------------------------------------------
    # Actual mesh spacing
    # --------------------------------------------------

    hx = lx / nx
    hy = ly / ny
    hz = lz / nz

    # --------------------------------------------------
    # Estimated mesh size
    # --------------------------------------------------

    estimated_nodes = (
        (nx + 1)
        * (ny + 1)
        * (nz + 1)
    )

    estimated_tets = (
        6
        * nx
        * ny
        * nz
    )

    # --------------------------------------------------
    # Feedback
    # --------------------------------------------------

    print(
        f"Geometry:"
        f" Lx={lx:.6e},"
        f" Ly={ly:.6e},"
        f" Lz={lz:.6e}"
    )

    print(
        f"Requested target size : "
        f"{target_size:.6e}"
    )

    print("\nDivisions:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")

    print("\nActual spacing:")
    print(f"  hx = {hx:.6e}")
    print(f"  hy = {hy:.6e}")
    print(f"  hz = {hz:.6e}")

    print("\nEstimated mesh:")
    print(
        f"  Nodes     : "
        f"{estimated_nodes:,}"
    )

    print(
        f"  Tetrahedra: "
        f"~{estimated_tets:,}"
    )

    # --------------------------------------------------
    # Check isotropy
    # --------------------------------------------------

    h_min = min(hx, hy, hz)
    h_max = max(hx, hy, hz)

    if h_min > 0:
        anisotropy_ratio = h_max / h_min
    else:
        anisotropy_ratio = float("inf")

    print(
        f"\nSpacing ratio h_max/h_min : "
        f"{anisotropy_ratio:.3f}"
    )

    if anisotropy_ratio > 1.25:
        print(
            "⚠ Mesh spacing is not perfectly isotropic."
        )
    else:
        print(
            "✓ Mesh spacing is approximately isotropic."
        )

    print(
        "======================================"
    )

    return nx, ny, nz
def _calculate_anisotropic_divisions(
    lx,
    ly,
    lz,
    bulk_size,
    fine_size,
    fine_radius,
    source_center,
    min_divisions=2,
    max_divisions=5000,
):
    """
    Calculate independent structured divisions for X, Y and Z.

    The bulk mesh is kept relatively coarse, while the region
    around the heat source receives finer resolution.

    This function only calculates the resolution.
    It does not create Gmsh geometry or generate the mesh.
    """

    print("\n======================================")
    print("=== ANISOTROPIC MESH RESOLUTION ===")
    print("======================================")

    if bulk_size <= 0:
        raise ValueError(
            "bulk_size must be greater than zero."
        )

    if fine_size <= 0:
        raise ValueError(
            "fine_size must be greater than zero."
        )

    if fine_size > bulk_size:
        raise ValueError(
            "fine_size cannot be larger than bulk_size."
        )

    if fine_radius <= 0:
        raise ValueError(
            "fine_radius must be greater than zero."
        )

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    dimensions = {
        "X": float(lx),
        "Y": float(ly),
        "Z": float(lz),
    }

    print("\nGeometry:")
    print(f"  Lx = {lx:.6e}")
    print(f"  Ly = {ly:.6e}")
    print(f"  Lz = {lz:.6e}")

    print("\nMesh sizes:")
    print(f"  Bulk size = {bulk_size:.6e}")
    print(f"  Fine size = {fine_size:.6e}")

    print(
        f"\nFine-region radius = "
        f"{fine_radius:.6e}"
    )

    print(
        f"Source center = "
        f"{source_center}"
    )

    # --------------------------------------------------
    # Basic bulk divisions
    # --------------------------------------------------

    nx_bulk = max(
        min_divisions,
        int(round(lx / bulk_size)),
    )

    ny_bulk = max(
        min_divisions,
        int(round(ly / bulk_size)),
    )

    nz_bulk = max(
        min_divisions,
        int(round(lz / bulk_size)),
    )

    # --------------------------------------------------
    # Fine-region divisions
    #
    # Only the dimensions that actually intersect
    # the fine region receive additional resolution.
    # --------------------------------------------------

    cx, cy, cz = [
        float(v)
        for v in source_center
    ]

    half_fine = fine_radius

    fine_x_min = cx - half_fine
    fine_x_max = cx + half_fine

    fine_y_min = cy - half_fine
    fine_y_max = cy + half_fine

    fine_z_min = cz - half_fine
    fine_z_max = cz + half_fine

    # --------------------------------------------------
    # Clamp fine region to geometry
    # --------------------------------------------------

    fine_x_min = max(
        -lx / 2.0,
        fine_x_min,
    )

    fine_x_max = min(
        lx / 2.0,
        fine_x_max,
    )

    fine_y_min = max(
        -ly / 2.0,
        fine_y_min,
    )

    fine_y_max = min(
        ly / 2.0,
        fine_y_max,
    )

    fine_z_min = max(
        -lz / 2.0,
        fine_z_min,
    )

    fine_z_max = min(
        lz / 2.0,
        fine_z_max,
    )

    fine_lx = max(
        0.0,
        fine_x_max - fine_x_min,
    )

    fine_ly = max(
        0.0,
        fine_y_max - fine_y_min,
    )

    fine_lz = max(
        0.0,
        fine_z_max - fine_z_min,
    )

    print("\nFine-region dimensions:")
    print(f"  X = {fine_lx:.6e}")
    print(f"  Y = {fine_ly:.6e}")
    print(f"  Z = {fine_lz:.6e}")

    # --------------------------------------------------
    # Additional divisions required by fine region
    # --------------------------------------------------

    extra_x = 0
    extra_y = 0
    extra_z = 0

    if fine_lx > 0:
        extra_x = int(
            round(fine_lx / fine_size)
        )

    if fine_ly > 0:
        extra_y = int(
            round(fine_ly / fine_size)
        )

    if fine_lz > 0:
        extra_z = int(
            round(fine_lz / fine_size)
        )

    # --------------------------------------------------
    # Final divisions
    # --------------------------------------------------

    nx = max(
        nx_bulk,
        extra_x,
        min_divisions,
    )

    ny = max(
        ny_bulk,
        extra_y,
        min_divisions,
    )

    nz = max(
        nz_bulk,
        extra_z,
        min_divisions,
    )

    # --------------------------------------------------
    # Safety limit
    # --------------------------------------------------

    nx = min(nx, max_divisions)
    ny = min(ny, max_divisions)
    nz = min(nz, max_divisions)

    # --------------------------------------------------
    # Resulting average spacing
    # --------------------------------------------------

    hx = lx / nx
    hy = ly / ny
    hz = lz / nz

    print("\n=== FINAL DIVISIONS ===")

    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")

    print("\nAverage spacing:")
    print(f"  hx = {hx:.6e}")
    print(f"  hy = {hy:.6e}")
    print(f"  hz = {hz:.6e}")

    # --------------------------------------------------
    # Estimated mesh size
    # --------------------------------------------------

    estimated_nodes = (
        (nx + 1)
        * (ny + 1)
        * (nz + 1)
    )

    estimated_tets = (
        6
        * nx
        * ny
        * nz
    )

    print("\nEstimated structured mesh:")
    print(
        f"  Nodes      : "
        f"{estimated_nodes:,}"
    )

    print(
        f"  Tetrahedra : "
        f"~{estimated_tets:,}"
    )

    if estimated_tets > 10_000_000:
        print(
            "\n⚠ WARNING:"
            " estimated mesh exceeds"
            " 10 million tetrahedra."
        )

    print(
        "\n✓ Anisotropic resolution calculated"
    )

    return nx, ny, nz
def _apply_structured_mesh(
    volume,
    nx,
    ny,
    nz,
):
    """
    Apply a structured transfinite mesh to a box volume.

    nx, ny, nz = number of intervals along X, Y, Z.
    """

    print("\n======================================")
    print("=== STRUCTURED MESH SETUP ===")
    print("======================================")

    print(f"Volume : {volume}")
    print(f"NX     : {nx}")
    print(f"NY     : {ny}")
    print(f"NZ     : {nz}")

    # ==================================================
    # 1. Get the six surfaces of the volume
    # ==================================================

    volume_boundary = gmsh.model.getBoundary(
        [(3, volume)],
        combined=False,
        oriented=False,
        recursive=False,
    )

    surfaces = [
        tag
        for dim, tag in volume_boundary
        if dim == 2
    ]

    print(
        f"\nBoundary surfaces : {len(surfaces)}"
    )

    if len(surfaces) != 6:
        raise RuntimeError(
            "Structured box requires exactly "
            f"6 surfaces, found {len(surfaces)}."
        )

    # ==================================================
    # 2. Get curves from the surfaces
    # ==================================================

    curve_set = set()

    for surface in surfaces:

        surface_boundary = gmsh.model.getBoundary(
            [(2, surface)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        for dim, tag in surface_boundary:

            if dim == 1:
                curve_set.add(tag)

    curves = sorted(curve_set)

    print(
        f"Boundary curves   : {len(curves)}"
    )

    if len(curves) != 12:
        raise RuntimeError(
            "Structured box requires exactly "
            f"12 curves, found {len(curves)}."
        )

    # ==================================================
    # 3. Apply transfinite curves
    # ==================================================

    print("\n=== TRANSFINITE CURVES ===")

    for curve in curves:

        endpoints = gmsh.model.getBoundary(
            [(1, curve)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        points = [
            tag
            for dim, tag in endpoints
            if dim == 0
        ]

        if len(points) != 2:
            raise RuntimeError(
                f"Curve {curve} has "
                f"{len(points)} endpoints."
            )

        x1, y1, z1 = _get_point_coordinates(
            points[0]
        )

        x2, y2, z2 = _get_point_coordinates(
            points[1]
        )

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        tol = 1e-12

        if dx > tol:
            n = nx + 1
            axis = "X"

        elif dy > tol:
            n = ny + 1
            axis = "Y"

        elif dz > tol:
            n = nz + 1
            axis = "Z"

        else:
            raise RuntimeError(
                f"Curve {curve} is degenerate."
            )

        gmsh.model.mesh.setTransfiniteCurve(
            curve,
            n,
        )

        print(
            f"Curve {curve:3d}: "
            f"{axis} → {n} points"
        )

    # ==================================================
    # 4. Apply transfinite surfaces
    # ==================================================

    print("\n=== TRANSFINITE SURFACES ===")

    for surface in surfaces:

        surface_boundary = gmsh.model.getBoundary(
            [(2, surface)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        surface_curves = [
            tag
            for dim, tag in surface_boundary
            if dim == 1
        ]

        if len(surface_curves) != 4:
            raise RuntimeError(
                f"Surface {surface} has "
                f"{len(surface_curves)} boundary curves; "
                "expected 4."
            )

        # Find the four corner points from the
        # four boundary curves.
        point_set = set()

        for curve in surface_curves:

            endpoints = gmsh.model.getBoundary(
                [(1, curve)],
                combined=False,
                oriented=False,
                recursive=False,
            )

            for dim, tag in endpoints:

                if dim == 0:
                    point_set.add(tag)

        points = sorted(point_set)

        if len(points) != 4:
            raise RuntimeError(
                f"Surface {surface} has "
                f"{len(points)} corner points; "
                "expected 4."
            )

        gmsh.model.mesh.setTransfiniteSurface(
            surface,
            "Left",
            points,
        )

        print(
            f"Surface {surface:3d}: "
            f"4 corners"
        )

    # ==================================================
    # 5. Find the eight volume corners
    # ==================================================

    print("\n=== VOLUME CORNERS ===")

    corner_set = set()

    for surface in surfaces:

        surface_boundary = gmsh.model.getBoundary(
            [(2, surface)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        surface_curves = [
            tag
            for dim, tag in surface_boundary
            if dim == 1
        ]

        for curve in surface_curves:

            endpoints = gmsh.model.getBoundary(
                [(1, curve)],
                combined=False,
                oriented=False,
                recursive=False,
            )

            for dim, tag in endpoints:

                if dim == 0:
                    corner_set.add(tag)

    corners = sorted(corner_set)

    print(
        f"Volume corners : {len(corners)}"
    )

    if len(corners) != 8:
        raise RuntimeError(
            "Structured box requires exactly "
            f"8 corners, found {len(corners)}."
        )

    for point in corners:

        x, y, z = _get_point_coordinates(
            point
        )

        print(
            f"  Point {point:3d}: "
            f"({x:.6e}, "
            f"{y:.6e}, "
            f"{z:.6e})"
        )

    # ==================================================
    # 6. Apply transfinite volume
    # ==================================================

    gmsh.model.mesh.setTransfiniteVolume(
        volume,
        corners,
    )

    print(
        "\n✓ Transfinite volume configured "
        "with 8 explicit corners"
    )

    # ==================================================
    # 7. Recombine surfaces/volume
    # ==================================================

    for surface in surfaces:

        gmsh.model.mesh.setRecombine(
            2,
            surface,
        )

    print(
        "✓ Surface recombination enabled"
    )

    # ==================================================
    # 8. Synchronize
    # ==================================================

    gmsh.model.geo.synchronize()

    print(
        "✓ Structured topology ready"
    )
def _get_point_coordinates(point_tag):
    """
    Return the XYZ coordinates of a Gmsh point.
    """

    data = gmsh.model.getValue(
        0,
        point_tag,
        [],
    )

    if len(data) != 3:
        raise RuntimeError(
            f"Invalid coordinates for point "
            f"{point_tag}: {data}"
        )

    return (
        float(data[0]),
        float(data[1]),
        float(data[2]),
    )
def _generate_mesh(
    volume,
    mesh_algorithm=1,
):
    """
    Generate the 3D Gmsh mesh.

    Parameters
    ----------
    volume : int
        Gmsh volume tag.

    mesh_algorithm : int
        Gmsh 3D meshing algorithm.

    Returns
    -------
    None
    """

    print("\n======================================")
    print("🔷 GENERATING 3D MESH")
    print("======================================")

    # --------------------------------------------------
    # Validate volume
    # --------------------------------------------------

    volumes = gmsh.model.getEntities(3)

    volume_tags = [
        tag
        for dim, tag in volumes
    ]

    if volume not in volume_tags:
        raise RuntimeError(
            f"Volume {volume} does not exist in "
            f"the Gmsh model."
        )

    print(
        f"Volume: {volume}"
    )

    print(
        f"3D entities: {volumes}"
    )

    # --------------------------------------------------
    # Select 3D meshing algorithm
    # --------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.Algorithm3D",
        mesh_algorithm,
    )

    print(
        f"3D algorithm: "
        f"{mesh_algorithm}"
    )

    # --------------------------------------------------
    # Generate tetrahedral mesh
    # --------------------------------------------------

    print(
        "\n🔷 Calling Gmsh 3D mesher..."
    )

    gmsh.model.mesh.generate(3)

    # --------------------------------------------------
    # Mesh diagnostics
    # --------------------------------------------------

    print(
        "\n=== GMSH MESH RESULT ==="
    )

    node_tags, node_coordinates, _ = (
        gmsh.model.mesh.getNodes()
    )

    print(
        f"Gmsh nodes       : "
        f"{len(node_tags):,}"
    )

    element_types, element_tags, _ = (
        gmsh.model.mesh.getElements(3)
    )

    total_3d_elements = sum(
        len(tags)
        for tags in element_tags
    )

    print(
        f"3D elements      : "
        f"{total_3d_elements:,}"
    )

    print(
        f"Element types    : "
        f"{element_types}"
    )

    # --------------------------------------------------
    # Element-type diagnostics
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Final check
    # --------------------------------------------------

    if len(node_tags) == 0:
        raise RuntimeError(
            "Gmsh generated zero mesh nodes."
        )

    if total_3d_elements == 0:
        raise RuntimeError(
            "Gmsh generated zero 3D elements."
        )

    print(
        "\n✓ Gmsh 3D mesh generated successfully"
    )
def _create_physical_groups(volume):
    """
    Create the physical groups required by DOLFINx.

    Parameters
    ----------
    volume : int
        Gmsh 3D volume tag.

    Returns
    -------
    dict
        Physical-group information.
    """

    print("\n======================================")
    print("=== PHYSICAL GROUPS ===")
    print("======================================")

    # --------------------------------------------------
    # Validate volume
    # --------------------------------------------------

    volumes = gmsh.model.getEntities(3)

    volume_tags = [
        tag
        for dim, tag in volumes
    ]

    if volume not in volume_tags:
        raise RuntimeError(
            f"Cannot create physical group: "
            f"volume {volume} does not exist."
        )

    # --------------------------------------------------
    # Remove existing physical groups
    #
    # This prevents duplicated groups if this function
    # is called after previous geometry operations.
    # --------------------------------------------------

    existing_groups = (
        gmsh.model.getPhysicalGroups()
    )

    if existing_groups:
        print(
            f"Existing physical groups: "
            f"{existing_groups}"
        )

        for dim, tag in existing_groups:
            gmsh.model.removePhysicalGroups(
                [(dim, tag)]
            )

        print(
            "✓ Existing physical groups removed"
        )

    # --------------------------------------------------
    # Physical volume
    # --------------------------------------------------

    physical_volume_tag = (
        gmsh.model.addPhysicalGroup(
            3,
            [volume],
            1,
        )
    )

    gmsh.model.setPhysicalName(
        3,
        physical_volume_tag,
        "Volume",
    )

    print(
        f"✓ Physical volume group: "
        f"{physical_volume_tag}"
    )

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    groups = gmsh.model.getPhysicalGroups()

    print(
        "\nFinal physical groups:"
    )

    for dim, tag in groups:

        name = gmsh.model.getPhysicalName(
            dim,
            tag,
        )

        entities = (
            gmsh.model.getEntitiesForPhysicalGroup(
                dim,
                tag,
            )
        )

        print(
            f"  dim={dim}, "
            f"tag={tag}, "
            f"name='{name}', "
            f"entities={entities}"
        )

    # --------------------------------------------------
    # Mandatory volume check
    # --------------------------------------------------

    volume_groups = [
        (dim, tag)
        for dim, tag in groups
        if dim == 3
    ]

    if not volume_groups:
        raise RuntimeError(
            "No physical volume group was created."
        )

    print(
        "\n✓ Physical volume group verified"
    )

    return {
        "volume": physical_volume_tag,
    }
def _generate_structured_mesh(
    volume,
    nx,
    ny,
    nz,
):
    """
    Generate a structured tetrahedral mesh for a simple
    box-like volume using Gmsh transfinite meshing.

    Parameters
    ----------
    volume : int
        Gmsh volume tag.

    nx, ny, nz : int
        Number of subdivisions along X, Y and Z.

    Returns
    -------
    tuple
        (nx, ny, nz)
    """

    print("\n======================================")
    print("🔷 STRUCTURED MESH GENERATION")
    print("======================================")

    # --------------------------------------------------
    # Validate subdivision counts
    # --------------------------------------------------

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)

    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(
            "Mesh subdivisions must be >= 1."
        )

    print(
        f"Subdivisions:"
        f"  X = {nx}"
        f"  Y = {ny}"
        f"  Z = {nz}"
    )

    # --------------------------------------------------
    # Validate volume
    # --------------------------------------------------

    volumes = gmsh.model.getEntities(3)

    if (3, volume) not in volumes:
        raise RuntimeError(
            f"Volume {volume} does not exist."
        )

    # --------------------------------------------------
    # Get boundary surfaces
    # --------------------------------------------------

    boundary = gmsh.model.getBoundary(
        [(3, volume)],
        oriented=False,
        recursive=False,
    )

    surfaces = [
        tag
        for dim, tag in boundary
        if dim == 2
    ]

    if len(surfaces) != 6:
        raise RuntimeError(
            "Structured meshing requires a six-face "
            f"box-like volume. Found {len(surfaces)} "
            "boundary surfaces."
        )

    print(
        f"Boundary surfaces: {len(surfaces)}"
    )

    # --------------------------------------------------
    # Get curves belonging to the surfaces
    # --------------------------------------------------

    curves = set()

    for surface in surfaces:

        surface_boundary = gmsh.model.getBoundary(
            [(2, surface)],
            oriented=False,
            recursive=False,
        )

        for dim, tag in surface_boundary:

            if dim == 1:
                curves.add(tag)

    curves = sorted(curves)

    if len(curves) != 12:
        raise RuntimeError(
            "Structured meshing requires a box-like "
            f"volume with 12 boundary curves. "
            f"Found {len(curves)}."
        )

    print(
        f"Boundary curves: {len(curves)}"
    )

    # --------------------------------------------------
    # Classify curves by their geometric direction
    # --------------------------------------------------

    x_curves = []
    y_curves = []
    z_curves = []

    tolerance = 1e-9

    for curve in curves:

        bbox = gmsh.model.getBoundingBox(
            1,
            curve,
        )

        xmin, ymin, zmin, xmax, ymax, zmax = bbox

        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        dz = abs(zmax - zmin)

        if dx > tolerance and dx >= dy and dx >= dz:

            x_curves.append(curve)

        elif dy > tolerance and dy >= dx and dy >= dz:

            y_curves.append(curve)

        elif dz > tolerance:

            z_curves.append(curve)

        else:

            raise RuntimeError(
                f"Could not determine direction "
                f"of curve {curve}."
            )

    print(
        f"X-directed curves: {len(x_curves)}"
    )

    print(
        f"Y-directed curves: {len(y_curves)}"
    )

    print(
        f"Z-directed curves: {len(z_curves)}"
    )

    if (
        len(x_curves) != 4
        or len(y_curves) != 4
        or len(z_curves) != 4
    ):
        raise RuntimeError(
            "Unexpected box topology. "
            "Expected 4 X, 4 Y and 4 Z curves."
        )

    # --------------------------------------------------
    # Transfinite curves
    # --------------------------------------------------

    print("\nApplying transfinite curve divisions...")

    for curve in x_curves:

        gmsh.model.mesh.setTransfiniteCurve(
            curve,
            nx + 1,
        )

    for curve in y_curves:

        gmsh.model.mesh.setTransfiniteCurve(
            curve,
            ny + 1,
        )

    for curve in z_curves:

        gmsh.model.mesh.setTransfiniteCurve(
            curve,
            nz + 1,
        )

    print("✓ X curves configured")
    print("✓ Y curves configured")
    print("✓ Z curves configured")

    # --------------------------------------------------
    # Transfinite surfaces
    # --------------------------------------------------

    print("\nApplying transfinite surfaces...")

    for surface in surfaces:

        gmsh.model.mesh.setTransfiniteSurface(
            surface
        )

    print(
        f"✓ {len(surfaces)} surfaces configured"
    )

    # --------------------------------------------------
    # Transfinite volume
    # --------------------------------------------------

    print("\nApplying transfinite volume...")

    gmsh.model.mesh.setTransfiniteVolume(
        volume
    )

    print(
        f"✓ Volume {volume} configured"
    )

    # --------------------------------------------------
    # Tetrahedral recombination
    # --------------------------------------------------

    # Keep the mesh tetrahedral.
    # Do NOT recombine surfaces into quads.
    #
    # This is important because DOLFINx expects the
    # final volume mesh to contain tetrahedral cells.

    print(
        "\n✓ Tetrahedral structured mesh configured"
    )

    # --------------------------------------------------
    # Generate 3D mesh
    # --------------------------------------------------

    print(
        "\n🔷 Generating tetrahedral mesh..."
    )

    gmsh.model.mesh.generate(3)

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

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

    print(
        "\n=== GMSH MESH RESULT ==="
    )

    print(
        f"Gmsh nodes       : "
        f"{len(node_tags)}"
    )

    print(
        f"3D elements      : "
        f"{total_3d_elements}"
    )

    print(
        f"Element types    : "
        f"{element_types}"
    )

    # --------------------------------------------------
    # Sanity check
    # --------------------------------------------------

    if len(node_tags) == 0:
        raise RuntimeError(
            "Gmsh generated zero nodes."
        )

    if total_3d_elements == 0:
        raise RuntimeError(
            "Gmsh generated zero 3D elements."
        )

    if 4 not in element_types:
        raise RuntimeError(
            "No tetrahedral elements "
            "(Gmsh type 4) were generated."
        )

    print(
        "✓ Valid tetrahedral volume mesh"
    )

    return nx, ny, nz

def _calculate_mesh_resolution(
    lx,
    ly,
    lz,
    target_size,
    min_divisions=1,
    max_divisions=500,
):
    """
    Calculate structured-mesh subdivisions from geometry dimensions.

    Parameters
    ----------
    lx, ly, lz : float
        Geometry dimensions in Gmsh/STL units.

    target_size : float
        Desired approximate element size in the same units
        as lx, ly and lz.

    min_divisions : int
        Minimum number of divisions per direction.

    max_divisions : int
        Safety limit for divisions per direction.

    Returns
    -------
    tuple
        (nx, ny, nz, hx, hy, hz)
    """

    print("\n======================================")
    print("🔷 MESH RESOLUTION")
    print("======================================")

    lx = float(lx)
    ly = float(ly)
    lz = float(lz)
    target_size = float(target_size)

    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(
            "Geometry dimensions must be positive."
        )

    if target_size <= 0:
        raise ValueError(
            "Target mesh size must be positive."
        )

    # --------------------------------------------------
    # Calculate number of divisions
    # --------------------------------------------------

    nx = max(
        min_divisions,
        int(round(lx / target_size)),
    )

    ny = max(
        min_divisions,
        int(round(ly / target_size)),
    )

    nz = max(
        min_divisions,
        int(round(lz / target_size)),
    )

    # --------------------------------------------------
    # Apply safety limits
    # --------------------------------------------------

    nx = min(nx, max_divisions)
    ny = min(ny, max_divisions)
    nz = min(nz, max_divisions)

    # --------------------------------------------------
    # Actual resulting element dimensions
    # --------------------------------------------------

    hx = lx / nx
    hy = ly / ny
    hz = lz / nz

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print("\nGeometry dimensions:")
    print(f"  Lx = {lx:.6e}")
    print(f"  Ly = {ly:.6e}")
    print(f"  Lz = {lz:.6e}")

    print(
        f"\nRequested target size : "
        f"{target_size:.6e}"
    )

    print("\nCalculated divisions:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")

    print("\nActual structured spacing:")
    print(f"  hx = {hx:.6e}")
    print(f"  hy = {hy:.6e}")
    print(f"  hz = {hz:.6e}")

    print("\nAspect ratio:")
    print(
        f"  hx : hy : hz = "
        f"{hx:.3e} : "
        f"{hy:.3e} : "
        f"{hz:.3e}"
    )

    # --------------------------------------------------
    # Estimated structured grid size
    # --------------------------------------------------

    estimated_nodes = (
        (nx + 1)
        * (ny + 1)
        * (nz + 1)
    )

    # A structured hexahedral grid contains
    # nx * ny * nz cells.
    #
    # Each hexahedron is split into 6 tetrahedra
    # by Gmsh's transfinite tetrahedralization.

    estimated_tets = (
        6
        * nx
        * ny
        * nz
    )

    print("\nEstimated mesh size:")
    print(
        f"  Grid nodes      : "
        f"{estimated_nodes:,}"
    )

    print(
        f"  Tetrahedra      : "
        f"{estimated_tets:,}"
    )

    print(
        "======================================"
    )

    return (
        nx,
        ny,
        nz,
        hx,
        hy,
        hz,
    )
def _select_target_mesh_size(
    lx,
    ly,
    lz,
    lc_min,
    lc_max,
    min_elements_thickness=4,
    max_elements_thickness=20,
):
    """
    Select a base mesh size for a simple geometry.

    The smallest geometric dimension controls the resolution.
    This avoids creating an unnecessarily dense mesh along
    very long dimensions.

    Parameters
    ----------
    lx, ly, lz : float
        Geometry dimensions in Gmsh/STL units.

    lc_min, lc_max : float
        Allowed mesh-size limits in the same units.

    min_elements_thickness : int
        Minimum number of elements across the smallest
        geometric dimension.

    max_elements_thickness : int
        Maximum number of elements across the smallest
        geometric dimension.

    Returns
    -------
    float
        Selected target mesh size.
    """

    print("\n======================================")
    print("🔷 TARGET MESH SIZE SELECTION")
    print("======================================")

    lx = float(lx)
    ly = float(ly)
    lz = float(lz)

    lc_min = float(lc_min)
    lc_max = float(lc_max)

    if lc_min <= 0:
        raise ValueError(
            "lc_min must be positive."
        )

    if lc_max <= 0:
        raise ValueError(
            "lc_max must be positive."
        )

    if lc_min > lc_max:
        raise ValueError(
            "lc_min cannot be greater than lc_max."
        )

    # --------------------------------------------------
    # Identify smallest dimension
    # --------------------------------------------------

    dimensions = {
        "X": lx,
        "Y": ly,
        "Z": lz,
    }

    smallest_axis = min(
        dimensions,
        key=dimensions.get,
    )

    smallest_dimension = dimensions[
        smallest_axis
    ]

    print("\nGeometry:")
    print(f"  X = {lx:.6e}")
    print(f"  Y = {ly:.6e}")
    print(f"  Z = {lz:.6e}")

    print(
        f"\nSmallest dimension : "
        f"{smallest_axis}"
    )

    print(
        f"Smallest size      : "
        f"{smallest_dimension:.6e}"
    )

    # --------------------------------------------------
    # Determine size from thickness
    # --------------------------------------------------
    #
    # We want at least a few elements across the
    # smallest dimension.
    #
    # Example:
    #
    # thickness = 2.5 mm
    # minimum elements = 4
    #
    # target size = 2.5 / 4
    #              = 0.625 mm
    #
    # This is then constrained by lc_min/lc_max.

    size_from_minimum = (
        smallest_dimension
        / min_elements_thickness
    )

    size_from_maximum = (
        smallest_dimension
        / max_elements_thickness
    )

    print("\nThickness-based limits:")

    print(
        f"  Minimum-resolution size : "
        f"{size_from_minimum:.6e}"
    )

    print(
        f"  Maximum-resolution size : "
        f"{size_from_maximum:.6e}"
    )

    # --------------------------------------------------
    # Select size
    # --------------------------------------------------

    target_size = size_from_minimum

    # Respect global Gmsh limits.

    target_size = min(
        target_size,
        lc_max,
    )

    target_size = max(
        target_size,
        lc_min,
    )

    # --------------------------------------------------
    # Final diagnostic
    # --------------------------------------------------

    elements_across_smallest = (
        smallest_dimension
        / target_size
    )

    print("\nMesh-size limits:")
    print(
        f"  lc_min = {lc_min:.6e}"
    )

    print(
        f"  lc_max = {lc_max:.6e}"
    )

    print("\nSelected target size:")
    print(
        f"  target_size = "
        f"{target_size:.6e}"
    )

    print(
        f"\nElements across "
        f"{smallest_axis}: "
        f"{elements_across_smallest:.2f}"
    )

    print(
        "======================================"
    )

    return target_size
def _apply_source_refinement(
    source,
    scale,
    surface_point,
    fine_size,
    fine_radius,
):
    """
    Create a Gmsh mesh-size field centered on the illuminated
    surface point.

    All Gmsh coordinates are in STL/Gmsh units.
    Source radius is provided in meters.
    """

    if source is None:
        print("\n=== SOURCE REFINEMENT ===")
        print("No source configured.")
        return None

    if surface_point is None:
        raise RuntimeError(
            "Cannot create source refinement: "
            "illuminated surface point is undefined."
        )

    radius_gmsh = float(source["radius"]) / float(scale)
    fine_radius = float(fine_radius)
    fine_size = float(fine_size)

    if radius_gmsh <= 0:
        raise ValueError("Source radius must be > 0.")

    if fine_radius <= 0:
        raise ValueError("Fine refinement radius must be > 0.")

    if fine_size <= 0:
        raise ValueError("Fine mesh size must be > 0.")

    print("\n======================================")
    print("=== SOURCE MESH REFINEMENT ===")
    print("======================================")

    print(f"Source center       : {source['center']}")
    print(f"Source radius (m)   : {source['radius']:.6e}")
    print(f"Source radius Gmsh  : {radius_gmsh:.6e}")
    print(f"Surface point       : {surface_point}")
    print(f"Fine radius         : {fine_radius:.6e}")
    print(f"Fine mesh size      : {fine_size:.6e}")

    field = gmsh.model.mesh.field.add("Distance")

    gmsh.model.mesh.field.setNumbers(
        field,
        "NodesList",
        [],
    )

    # Use a geometrical point at the illuminated location.
    point_tag = gmsh.model.geo.addPoint(
        float(surface_point[0]),
        float(surface_point[1]),
        float(surface_point[2]),
        fine_size,
    )

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.setNumbers(
        field,
        "NodesList",
        [point_tag],
    )

    threshold = gmsh.model.mesh.field.add("Threshold")

    gmsh.model.mesh.field.setNumber(
        threshold,
        "InField",
        field,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMin",
        fine_size,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMax",
        float(fine_size) * 5.0,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMin",
        0.0,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMax",
        fine_radius,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "StopAtDistMax",
        1,
    )

    gmsh.model.mesh.field.setAsBackgroundMesh(
        threshold
    )

    print(f"✓ Refinement point created: {point_tag}")
    print(f"✓ Distance field created: {field}")
    print(f"✓ Threshold field created: {threshold}")
    print("✓ Source refinement enabled")

    return threshold


def _create_physical_groups(volume):
    """
    Create and verify the physical volume group required by
    DOLFINx.
    """

    print("\n======================================")
    print("=== PHYSICAL GROUPS ===")
    print("======================================")

    volumes = gmsh.model.getEntities(3)

    if (3, volume) not in volumes:
        raise RuntimeError(
            f"Cannot create physical group: "
            f"volume {volume} does not exist."
        )

    # Remove stale groups.
    existing = gmsh.model.getPhysicalGroups()

    if existing:
        print(
            f"Removing existing physical groups: "
            f"{existing}"
        )

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

    print(
        f"✓ Physical volume group: "
        f"{physical_tag}"
    )

    groups = gmsh.model.getPhysicalGroups()

    if (3, physical_tag) not in groups:
        raise RuntimeError(
            "Physical volume group creation failed."
        )

    entities = gmsh.model.getEntitiesForPhysicalGroup(
        3,
        physical_tag,
    )

    if volume not in entities:
        raise RuntimeError(
            "Physical group does not contain "
            f"volume {volume}."
        )

    print(
        f"✓ Verified volume {volume} "
        f"belongs to physical group {physical_tag}"
    )

    return {
        "volume": physical_tag,
    }


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
    Generate a structured tetrahedral volume mesh for a
    simple STL-derived geometry.

    The STL is used only to determine the bounding box.
    The computational geometry is reconstructed as an OCC box.
    """

    del feature_angle

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    xdmf_path = os.path.join(
        output_dir,
        output_filename,
    )

    print("\n======================================")
    print("🔷 SMART VOLUME MESH")
    print("======================================")

    # --------------------------------------------------
    # Units
    # --------------------------------------------------

    scale = _unit_scale(unit)

    print("\n=== UNIT SYSTEM ===")
    print(f"STL/Gmsh unit : {unit}")
    print(f"STL → meters  : {scale}")

    # --------------------------------------------------
    # Imported STL
    # --------------------------------------------------

    surfaces = gmsh.model.getEntities(2)

    if not surfaces:
        raise RuntimeError(
            "No STL surfaces found in Gmsh model."
        )

    print("\n=== IMPORTED STL ===")
    print(f"Surfaces: {surfaces}")

    # --------------------------------------------------
    # Bounding box
    # --------------------------------------------------

    bbox = gmsh.model.getBoundingBox(
        -1,
        -1,
    )

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

    print("\n=== GEOMETRY ===")
    print(f"X: {xmin:.6e} -> {xmax:.6e}")
    print(f"Y: {ymin:.6e} -> {ymax:.6e}")
    print(f"Z: {zmin:.6e} -> {zmax:.6e}")
    print(f"Lx = {Lx:.6e} {unit}")
    print(f"Ly = {Ly:.6e} {unit}")
    print(f"Lz = {Lz:.6e} {unit}")

    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise RuntimeError(
            "Invalid STL bounding box."
        )

    # --------------------------------------------------
    # Source
    # --------------------------------------------------

    source = _get_source_parameters(
        source_config
    )

    center_gmsh = None
    illuminated_face = None
    surface_point = None

    if source is not None:

        center_gmsh = [
            float(v) / scale
            for v in source["center"]
        ]

        (
            illuminated_face,
            surface_point,
        ) = _find_illuminated_face(
            bbox,
            center_gmsh,
            source["direction"],
        )

        print("\n=== SOURCE ===")
        print(
            f"Type                : "
            f"{source['type']}"
        )
        print(
            f"Center (m)          : "
            f"{source['center']}"
        )
        print(
            f"Center (Gmsh)       : "
            f"{center_gmsh}"
        )
        print(
            f"Radius (m)          : "
            f"{source['radius']:.6e}"
        )
        print(
            f"Direction            : "
            f"{source['direction']}"
        )
        print(
            f"Illuminated face    : "
            f"{illuminated_face}"
        )
        print(
            f"Surface point       : "
            f"{surface_point}"
        )

    else:
        print("\n=== SOURCE ===")
        print("No Gaussian source configured.")

    # --------------------------------------------------
    # Resolution
    # --------------------------------------------------

    resolution = _calculate_resolution(
        Lx,
        Ly,
        Lz,
        float(lc_min),
        float(lc_max),
        source,
        scale,
    )

    nx = resolution["nx_bulk"]
    ny = resolution["ny_bulk"]
    nz = resolution["nz"]

    fine_xy = resolution["fine_xy"]

    print("\n=== FINAL RESOLUTION ===")
    print(f"NX = {nx}")
    print(f"NY = {ny}")
    print(f"NZ = {nz}")
    print(
        f"Estimated cells = "
        f"{nx * ny * nz:,}"
    )

    # --------------------------------------------------
    # Create computational volume
    # --------------------------------------------------

    print("\n=== BUILDING SIMPLE VOLUME ===")
    print("Using STL bounding box.")

    volume = _create_structured_box(
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
    )

    print(
        f"✓ Computational volume: {volume}"
    )

    # --------------------------------------------------
    # Source refinement
    # --------------------------------------------------

    if source is not None:

        radius_gmsh = (
            source["radius"] / scale
        )

        fine_radius = max(
            1.5 * radius_gmsh,
            3.0 * fine_xy,
        )

        _apply_source_refinement(
            source=source,
            scale=scale,
            surface_point=surface_point,
            fine_size=fine_xy,
            fine_radius=fine_radius,
        )

    # --------------------------------------------------
    # Structured mesh
    # --------------------------------------------------

    _apply_structured_mesh(
        volume,
        nx,
        ny,
        nz,
    )

    # --------------------------------------------------
    # Physical groups
    # --------------------------------------------------

    physical_groups = _create_physical_groups(
        volume
    )

    # --------------------------------------------------
    # Generate mesh
    # --------------------------------------------------

    _generate_mesh(
        volume,
        mesh_algorithm=1,
    )

    # --------------------------------------------------
    # Convert
    # --------------------------------------------------

    print(
        "\n🔄 Converting Gmsh → DOLFINx..."
    )

    mesh = convert_model_to_mesh(
        unit
    )

    print(
        "✓ DOLFINx conversion successful"
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print_mesh_info(
        mesh
    )

    # --------------------------------------------------
    # Write
    # --------------------------------------------------

    print("\n💾 Writing XDMF:")
    print(xdmf_path)

    write_xdmf(
        mesh,
        xdmf_path,
    )

    print(
        "\n======================================"
    )
    print(
        "✅ SMART VOLUME MESH CREATED"
    )
    print(
        "======================================"
    )

    return xdmf_path
def _generate_mesh(
    volume,
    mesh_algorithm=1,
):
    """
    Generate the final 3D tetrahedral mesh.

    The structured/transfinite constraints must already have
    been applied before this function is called.
    """

    print("\n======================================")
    print("🔷 GENERATING 3D STRUCTURED MESH")
    print("======================================")

    # --------------------------------------------------
    # Validate volume
    # --------------------------------------------------

    volumes = gmsh.model.getEntities(3)

    if (3, volume) not in volumes:
        raise RuntimeError(
            f"Cannot generate mesh: "
            f"volume {volume} does not exist."
        )

    print(f"Volume              : {volume}")
    print(f"3D entities          : {volumes}")

    # --------------------------------------------------
    # Gmsh options
    # --------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.Algorithm3D",
        int(mesh_algorithm),
    )

    # Keep tetrahedral elements.
    gmsh.option.setNumber(
        "Mesh.RecombineAll",
        0,
    )

    # Generate only after all transfinite constraints
    # and physical groups have been installed.
    print("\nCalling Gmsh 3D mesher...")

    gmsh.model.mesh.generate(3)

    # --------------------------------------------------
    # Retrieve mesh
    # --------------------------------------------------

    node_tags, node_coordinates, _ = (
        gmsh.model.mesh.getNodes()
    )

    element_types, element_tags, element_nodes = (
        gmsh.model.mesh.getElements(3)
    )

    total_elements = sum(
        len(tags)
        for tags in element_tags
    )

    print("\n=== GMSH MESH RESULT ===")
    print(
        f"Gmsh nodes       : "
        f"{len(node_tags):,}"
    )
    print(
        f"3D elements      : "
        f"{total_elements:,}"
    )
    print(
        f"Element types    : "
        f"{element_types}"
    )

    # --------------------------------------------------
    # Element diagnostics
    # --------------------------------------------------

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

        # Gmsh type 4 = 4-node linear tetrahedron
        if element_type == 4:
            tetrahedral_count += len(tags)

    # --------------------------------------------------
    # Mandatory checks
    # --------------------------------------------------

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
        f"\n✓ Linear tetrahedra : "
        f"{tetrahedral_count:,}"
    )

    print(
        "\n✓ 3D structured tetrahedral "
        "mesh generated successfully"
    )

    return {
        "nodes": len(node_tags),
        "elements": total_elements,
        "tetrahedra": tetrahedral_count,
        "element_types": list(element_types),
    }
