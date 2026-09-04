from dolfinx import fem
from dolfinx import mesh
from ufl import Measure
import numpy as np


# ============================================================
# INITIALIZE LAMP
# ============================================================

def initialize_lamp(heat_eq, source):
    """
    Initialize a lamp source.

    The lamp is represented as an incident irradiance [W/m²]
    applied to the illuminated boundary of the mesh.

    For a volume mesh:
        - Detect illuminated exterior facets.
        - Apply the irradiance on those facets.

    For a surface mesh:
        - Detect illuminated surface cells.

    The absorbed thermal flux is:

        q_abs = irradiance * absorptivity

    where:
        irradiance   : incident radiation [W/m²]
        absorptivity : fraction converted into heat [-]
    """

    # --------------------------------------------------
    # Heat flux
    # --------------------------------------------------

    heat_eq.q_lamp = fem.Constant(
        heat_eq.mesh,
        0.0,
    )

    heat_eq.ds_lamp = None
    heat_eq.illuminated_area = None

    # --------------------------------------------------
    # Lamp geometry
    # --------------------------------------------------

    source_position, light_direction = parse_lamp_geometry(
        source
    )

    topo_dim = heat_eq.mesh.topology.dim

    # --------------------------------------------------
    # Detect illuminated entities
    # --------------------------------------------------

    if topo_dim == 3:

        (
            heat_eq.boundary_facets,
            heat_eq.illuminated_area,
        ) = find_volume_lamp_facets(
            heat_eq.mesh,
            source_position,
            light_direction,
        )

        entity_dim = topo_dim - 1

    elif topo_dim == 2:

        (
            heat_eq.boundary_facets,
            heat_eq.illuminated_area,
        ) = find_surface_lamp_facets(
            heat_eq.mesh,
            light_direction,
        )

        entity_dim = topo_dim

    else:

        raise ValueError(
            f"Unsupported mesh topology ({topo_dim})."
        )

    if len(heat_eq.boundary_facets) == 0:

        raise ValueError(
            "No illuminated facets found."
        )

    # --------------------------------------------------
    # Mesh tags
    # --------------------------------------------------

    facet_values = np.ones(
        len(heat_eq.boundary_facets),
        dtype=np.int32,
    )

    entities = np.asarray(
        heat_eq.boundary_facets,
        dtype=np.int32,
    )

    order = np.argsort(entities)

    entities = entities[order]
    facet_values = facet_values[order]

    heat_eq.facet_tags = mesh.meshtags(
        heat_eq.mesh,
        entity_dim,
        entities,
        facet_values,
    )

    # --------------------------------------------------
    # Integration measure
    # --------------------------------------------------

    if topo_dim == 3:

        heat_eq.ds_lamp = Measure(
            "ds",
            domain=heat_eq.mesh,
            subdomain_data=heat_eq.facet_tags,
        )

    else:

        heat_eq.ds_lamp = Measure(
            "dx",
            domain=heat_eq.mesh,
            subdomain_data=heat_eq.facet_tags,
        )

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    print(
        f"💡 Illuminated entities : "
        f"{len(heat_eq.boundary_facets)}"
    )

    print(
        f"💡 Illuminated area     : "
        f"{heat_eq.illuminated_area:.6e} m²"
    )


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def facet_normal(points):
    """
    Compute the unit normal of a triangular facet.
    """

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    n = np.cross(v1, v2)

    norm = np.linalg.norm(n)

    if norm == 0:
        return None

    return n / norm


def triangle_area(points):
    """
    Compute the area of a triangular facet.
    """

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    return 0.5 * np.linalg.norm(
        np.cross(v1, v2)
    )


# ============================================================
# FIND TOP FACETS
# ============================================================

def find_top_facets(
    domain,
    light_direction=np.array(
        [0.0, 0.0, 1.0]
    ),
):
    """
    Find exterior facets illuminated by a parallel light source.
    """

    fdim = domain.topology.dim - 1

    domain.topology.create_connectivity(
        fdim,
        0,
    )

    domain.topology.create_connectivity(
        fdim,
        domain.topology.dim,
    )

    domain.topology.create_connectivity(
        domain.topology.dim,
        fdim,
    )

    facets = mesh.exterior_facet_indices(
        domain.topology
    )

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(
        fdim,
        0,
    )

    mesh_center = x.mean(axis=0)

    illuminated = []

    area = 0.0

    light_direction = np.asarray(
        light_direction,
        dtype=float,
    )

    norm = np.linalg.norm(
        light_direction
    )

    if norm == 0:

        raise ValueError(
            "Light direction cannot be the zero vector."
        )

    light_direction /= norm

    for facet in facets:

        vertices = connectivity.links(facet)

        if len(vertices) != 3:
            continue

        points = x[vertices]

        normal = facet_normal(points)

        if normal is None:
            continue

        center = points.mean(axis=0)

        # Orient normal outwards
        if np.dot(
            normal,
            center - mesh_center,
        ) < 0:

            normal *= -1.0

        if np.dot(
            normal,
            light_direction,
        ) > 0:

            illuminated.append(facet)

            area += triangle_area(points)

    return (
        np.asarray(
            illuminated,
            dtype=np.int32,
        ),
        area,
    )


# ============================================================
# UPDATE LAMP SOURCE
# ============================================================

def update_lamp_source(
    heat_eq,
    source,
    t,
):
    """
    Update the absorbed lamp heat flux.

    Incident radiation:

        irradiance [W/m²]

    Absorbed radiation:

        q_abs [W/m²]
             = irradiance * absorptivity
    """

    t0 = float(
        source["t_start"]
    )

    t1 = float(
        source["t_end"]
    )

    # --------------------------------------------------
    # Lamp OFF
    # --------------------------------------------------

    if not (t0 <= t <= t1):

        heat_eq.q_lamp.value = 0.0

        if heat_eq.debug:

            print(
                f"💡 Lamp OFF "
                f"(t={t:.3f}s)"
            )

        return

    # --------------------------------------------------
    # Incident irradiance
    # --------------------------------------------------

    irradiance = float(
        source["irradiance"]
    )

    # --------------------------------------------------
    # Absorptivity
    # --------------------------------------------------

    absorptivity = float(
        source.get(
            "absorptivity",
            1.0,
        )
    )

    if not 0.0 <= absorptivity <= 1.0:

        raise ValueError(
            "Lamp absorptivity must be "
            "between 0 and 1."
        )

    # --------------------------------------------------
    # Absorbed heat flux
    # --------------------------------------------------

    heat_eq.q_lamp.value = (
        irradiance * absorptivity
    )

    if heat_eq.debug:

        print(
            f"💡 Lamp ON "
            f"(t={t:.3f}s)"
        )

        print(
            f"   Irradiance   = "
            f"{irradiance:.6e} W/m²"
        )

        print(
            f"   Absorptivity = "
            f"{absorptivity:.6f}"
        )

        print(
            f"   Absorbed flux = "
            f"{heat_eq.q_lamp.value:.6e} W/m²"
        )


# ============================================================
# FIND VOLUME LAMP FACETS
# ============================================================

def find_volume_lamp_facets(
    domain,
    source_position,
    light_direction,
):
    """
    Find illuminated boundary facets of a volume mesh.

    Parameters
    ----------
    source_position : np.ndarray
        Lamp position.

        If any coordinate is ±inf, the source is treated
        as infinitely far away and the rays are parallel.

    light_direction : np.ndarray
        Direction of light propagation.
    """

    fdim = domain.topology.dim - 1

    domain.topology.create_connectivity(
        fdim,
        0,
    )

    domain.topology.create_connectivity(
        fdim,
        domain.topology.dim,
    )

    domain.topology.create_connectivity(
        domain.topology.dim,
        fdim,
    )

    facets = mesh.exterior_facet_indices(
        domain.topology
    )

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(
        fdim,
        0,
    )

    mesh_center = x.mean(axis=0)

    infinite_source = np.any(
        np.isinf(source_position)
    )

    illuminated = []

    area = 0.0

    # Normalize light direction
    light_direction = np.asarray(
        light_direction,
        dtype=float,
    )

    norm = np.linalg.norm(
        light_direction
    )

    if norm == 0:

        raise ValueError(
            "Light direction cannot be the zero vector."
        )

    light_direction /= norm

    for facet in facets:

        vertices = connectivity.links(facet)

        if len(vertices) != 3:
            continue

        points = x[vertices]

        normal = facet_normal(points)

        if normal is None:
            continue

        center = points.mean(axis=0)

        # --------------------------------------------------
        # Orient normal outwards
        # --------------------------------------------------

        if np.dot(
            normal,
            center - mesh_center,
        ) < 0:

            normal *= -1.0

        # --------------------------------------------------
        # Compute incoming light direction
        # --------------------------------------------------

        if infinite_source:

            ray = light_direction

        else:

            ray = (
                center - source_position
            )

            ray_norm = np.linalg.norm(ray)

            if ray_norm == 0:
                continue

            ray /= ray_norm

            # Ignore facets behind the lamp
            if np.dot(
                ray,
                light_direction,
            ) < 0:

                continue

        # --------------------------------------------------
        # Illuminated?
        # --------------------------------------------------

        if np.dot(
            normal,
            ray,
        ) > 0:

            illuminated.append(facet)

            area += triangle_area(points)

    return (
        np.asarray(
            illuminated,
            dtype=np.int32,
        ),
        area,
    )


# ============================================================
# FIND SURFACE LAMP FACETS
# ============================================================

def find_surface_lamp_facets(
    domain,
    light_direction=np.array(
        [0.0, 0.0, 1.0]
    ),
):
    """
    Find illuminated cells of a 2D surface mesh.
    """

    tdim = domain.topology.dim

    domain.topology.create_connectivity(
        tdim,
        0,
    )

    cells = np.arange(
        domain.topology.index_map(
            tdim
        ).size_local,
        dtype=np.int32,
    )

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(
        tdim,
        0,
    )

    mesh_center = x.mean(
        axis=0
    )

    illuminated = []

    area = 0.0

    light_direction = np.asarray(
        light_direction,
        dtype=float,
    )

    norm = np.linalg.norm(
        light_direction
    )

    if norm == 0:

        raise ValueError(
            "Light direction cannot be the zero vector."
        )

    light_direction /= norm

    for cell in cells:

        vertices = connectivity.links(cell)

        if len(vertices) != 3:
            continue

        points = x[vertices]

        normal = facet_normal(points)

        if normal is None:
            continue

        center = points.mean(
            axis=0
        )

        # Orient normal consistently
        if np.dot(
            normal,
            center - mesh_center,
        ) < 0:

            normal *= -1.0

        if np.dot(
            normal,
            light_direction,
        ) > 0:

            illuminated.append(cell)

            area += triangle_area(points)

    return (
        np.asarray(
            illuminated,
            dtype=np.int32,
        ),
        area,
    )


# ============================================================
# PARSE LAMP GEOMETRY
# ============================================================

def parse_lamp_geometry(source):
    """
    Parse lamp position and light direction.

    Returns
    -------
    source_position : np.ndarray
        Lamp position. Infinite coordinates indicate
        an infinitely distant source.

    light_direction : np.ndarray
        Normalized direction of light propagation.
    """

    source_position = np.array(
        source.get(
            "source",
            [0.0, 0.0, np.inf],
        ),
        dtype=float,
    )

    light_direction = np.array(
        source.get(
            "direction",
            [0.0, 0.0, -1.0],
        ),
        dtype=float,
    )

    norm = np.linalg.norm(
        light_direction
    )

    if norm == 0:

        raise ValueError(
            "Lamp direction cannot be the zero vector."
        )

    light_direction /= norm

    return (
        source_position,
        light_direction,
    )
