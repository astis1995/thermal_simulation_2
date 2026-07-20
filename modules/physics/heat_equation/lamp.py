from dolfinx import fem
from dolfinx import fem, mesh
from ufl import Measure
import numpy as np

def initialize_lamp(heat_eq, source):
    """
    Initialize a lamp source.

    For volume meshes:
        - Detect illuminated boundary facets.

    For surface meshes:
        - Detect illuminated surface cells.

    Then:
        - Compute illuminated area.
        - Build the integration measure.
    """

    heat_eq.q_flux = fem.Constant(
        heat_eq.mesh,
        0.0,
    )

    heat_eq.ds_lamp = None
    heat_eq.illuminated_area = None

    source_position, light_direction = parse_lamp_geometry(source)

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
            source_position,
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
    # Measure
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

    print(f"💡 Illuminated entities : {len(heat_eq.boundary_facets)}")
    print(f"💡 Illuminated area     : {heat_eq.illuminated_area:.6e} m²")

from dolfinx import mesh
import numpy as np


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

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    return 0.5 * np.linalg.norm(
        np.cross(v1, v2)
    )


def find_top_facets(
    domain,
    light_direction=np.array([0.0, 0.0, 1.0])
):

    fdim = domain.topology.dim - 1

    domain.topology.create_connectivity(fdim, 0)
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, fdim)

    facets = mesh.exterior_facet_indices(domain.topology)

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(fdim, 0)

    mesh_center = x.mean(axis=0)

    illuminated = []

    area = 0.0

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
        if np.dot(normal, center - mesh_center) < 0:
            normal *= -1.0

        if np.dot(normal, light_direction) > 0:

            illuminated.append(facet)

            area += triangle_area(points)

    return (
        np.asarray(
            illuminated,
            dtype=np.int32,
        ),
        area,
    )

def update_lamp_source(heat_eq, source, t):
    """
    Update lamp boundary heat flux.
    """

    t0 = float(source["t_start"])
    t1 = float(source["t_end"])

    if not (t0 <= t <= t1):

        heat_eq.q_flux.value = 0.0

        if heat_eq.debug:
            print(f"💡 Lamp OFF (t={t:.3f}s)")

        return

    heat_eq.q_flux.value = float(source["power"])

    if heat_eq.debug:
        print(
            f"💡 Lamp ON (t={t:.3f}s)"
        )

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
        Lamp position. If any coordinate is ±inf, the source is treated
        as infinitely far away and rays are assumed parallel.

    light_direction : np.ndarray
        Main propagation direction of the light.
    """

    fdim = domain.topology.dim - 1

    domain.topology.create_connectivity(fdim, 0)
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, fdim)

    facets = mesh.exterior_facet_indices(domain.topology)

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(fdim, 0)

    mesh_center = x.mean(axis=0)

    infinite_source = np.any(np.isinf(source_position))

    illuminated = []

    area = 0.0

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
        if np.dot(normal, center - mesh_center) < 0:
            normal *= -1.0

        # --------------------------------------------
        # Compute incoming light direction
        # --------------------------------------------

        if infinite_source:

            ray = light_direction

        else:

            ray = center - source_position

            norm = np.linalg.norm(ray)

            if norm == 0:
                continue

            ray /= norm

            # Ignore facets behind the lamp
            if np.dot(ray, light_direction) < 0:
                continue

        # --------------------------------------------
        # Illuminated?
        # --------------------------------------------

        if np.dot(normal, ray) > 0:

            illuminated.append(facet)

            area += triangle_area(points)

    return (
        np.asarray(
            illuminated,
            dtype=np.int32,
        ),
        area,
    )


def find_surface_lamp_facets(
    domain,
    light_direction=np.array([0.0, 0.0, 1.0]),
):

    tdim = domain.topology.dim

    domain.topology.create_connectivity(tdim, 0)

    cells = np.arange(
        domain.topology.index_map(tdim).size_local,
        dtype=np.int32,
    )

    x = domain.geometry.x

    connectivity = domain.topology.connectivity(tdim, 0)

    mesh_center = x.mean(axis=0)

    illuminated = []

    area = 0.0

    for cell in cells:

        vertices = connectivity.links(cell)

        if len(vertices) != 3:
            continue

        points = x[vertices]

        normal = facet_normal(points)

        if normal is None:
            continue

        center = points.mean(axis=0)

        # Orient normal consistently
        if np.dot(normal, center - mesh_center) < 0:
            normal *= -1.0

        if np.dot(normal, light_direction) > 0:

            illuminated.append(cell)
            area += triangle_area(points)

    return (
        np.asarray(illuminated, dtype=np.int32),
        area,
    )


import numpy as np

def parse_lamp_geometry(source):
    """
    Returns
    -------
    source_position : np.ndarray
    light_direction : np.ndarray
    """

    # Default: sunlight coming from +Z
    source_position = np.array(
        source.get("source", [0.0, np.inf, 0.0]),
        dtype=float,
    )

    light_direction = np.array(
        source.get("direction", [0.0, 0.0, 1.0]),
        dtype=float,
    )

    norm = np.linalg.norm(light_direction)

    if norm == 0:
        raise ValueError(
            "Lamp direction cannot be the zero vector."
        )

    light_direction /= norm

    return source_position, light_direction
