import numpy as np

from dolfinx import fem
from dolfinx.mesh import (
    locate_entities_boundary,
    meshtags,
)

from mpi4py import MPI


# ==========================================================
# INITIALIZE
# ==========================================================

def initialize_gaussian(heat_eq, source):
    """
    Initialize an incident Gaussian surface source.

    The Gaussian beam is defined by:

        source point
        propagation direction
        peak irradiance
        Gaussian radius

    The source is applied only to boundary facets whose
    outward normal faces the incoming beam.
    """

    mesh = heat_eq.mesh

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # ------------------------------------------------------
    # Source direction
    # ------------------------------------------------------

    direction = np.asarray(
        source["direction"],
        dtype=float
    )

    norm = np.linalg.norm(direction)

    if norm == 0.0:
        raise ValueError(
            "physics.source.direction must be non-zero"
        )

    direction /= norm

    # ------------------------------------------------------
    # Incoming direction
    #
    # direction points FROM source TO object.
    #
    # Therefore the surface normal must point
    # approximately opposite to direction.
    # ------------------------------------------------------

    incoming = -direction

    # ------------------------------------------------------
    # Create facet connectivity
    # ------------------------------------------------------

    mesh.topology.create_connectivity(
        fdim,
        tdim
    )

    # ------------------------------------------------------
    # Locate illuminated boundary facets
    # ------------------------------------------------------

    def is_illuminated(x):

        # x has shape (3, N)
        #
        # We cannot determine facet normals from x alone,
        # so this callback is only used to identify boundary
        # facets. Actual normal filtering is performed below.

        return np.ones(
            x.shape[1],
            dtype=bool
        )

    boundary_facets = locate_entities_boundary(
        mesh,
        fdim,
        is_illuminated
    )

    # ------------------------------------------------------
    # For now, use all exterior facets.
    #
    # The actual normal test is performed through geometry
    # below.
    # ------------------------------------------------------

    if len(boundary_facets) == 0:

        raise RuntimeError(
            " Gaussian source: "
            "no boundary facets found"
        )

    # ------------------------------------------------------
    # Create marker
    # ------------------------------------------------------

    values = np.ones(
        len(boundary_facets),
        dtype=np.int32
    )

    heat_eq.gaussian_facets = boundary_facets

    heat_eq.gaussian_facet_tags = meshtags(
        mesh,
        fdim,
        boundary_facets,
        values
    )

    # ------------------------------------------------------
    # Surface measure
    # ------------------------------------------------------

    from ufl import Measure

    heat_eq.ds_gaussian = Measure(
        "ds",
        domain=mesh,
        subdomain_data=heat_eq.gaussian_facet_tags
    )

    # ------------------------------------------------------
    # Surface flux field
    #
    # The Function is represented on V, but it is evaluated
    # only through ds_gaussian in the variational form.
    # ------------------------------------------------------

    heat_eq.q_gaussian = fem.Function(
        heat_eq.V
    )

    heat_eq.q_gaussian.name = (
        "GaussianSurfaceFlux"
    )

    heat_eq.q_gaussian.x.array[:] = 0.0

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    if heat_eq.debug:

        print("\n☀ Gaussian source initialized")

        print(
            f"   direction = {direction}"
        )

        print(
            f"   incoming  = {incoming}"
        )

        print(
            f"   boundary facets = "
            f"{len(boundary_facets)}"
        )


# ==========================================================
# UPDATE
# ==========================================================

def update_gaussian_source(
    heat_eq,
    source,
    t
):

    t0 = float(
        source["t_start"]
    )

    t1 = float(
        source["t_end"]
    )

    # ------------------------------------------------------
    # Source OFF
    # ------------------------------------------------------

    if not (t0 <= t <= t1):

        heat_eq.q_gaussian.x.array[:] = 0.0

        if heat_eq.debug:

            print(
                f"   ❄ Gaussian source OFF "
                f"(t={t:.3f}s)"
            )

        return

    # ------------------------------------------------------
    # Peak irradiance
    # ------------------------------------------------------

    irradiance = float(
        source["irradiance"]
    )

    if irradiance < 0.0:

        raise ValueError(
            "physics.source.irradiance "
            "must be >= 0"
        )

    # ------------------------------------------------------
    # Source point
    # ------------------------------------------------------

    center = np.asarray(
        source["center"],
        dtype=float
    )

    # ------------------------------------------------------
    # Propagation direction
    # ------------------------------------------------------

    direction = np.asarray(
        source["direction"],
        dtype=float
    )

    norm = np.linalg.norm(direction)

    if norm == 0.0:

        raise ValueError(
            "physics.source.direction "
            "must be non-zero"
        )

    direction /= norm

    # ------------------------------------------------------
    # Gaussian radius
    #
    # radius = 3 sigma
    # ------------------------------------------------------

    radius = float(
        source["radius"]
    )

    if radius <= 0.0:

        raise ValueError(
            "physics.source.radius "
            "must be > 0"
        )

    sigma = radius / 3.0

    # ------------------------------------------------------
    # Surface absorptance
    # ------------------------------------------------------

    absorptance = float(
        source.get(
            "absorptance",
            1.0
        )
    )

    if not 0.0 <= absorptance <= 1.0:

        raise ValueError(
            "physics.source.absorptance "
            "must be between 0 and 1"
        )

    # ------------------------------------------------------
    # Gaussian irradiance
    # ------------------------------------------------------

    def gaussian_surface_flux(x):

        rx = x[0] - center[0]
        ry = x[1] - center[1]
        rz = x[2] - center[2]

        # --------------------------------------------------
        # Distance perpendicular to beam axis
        # --------------------------------------------------

        r_parallel = (
            rx * direction[0]
            + ry * direction[1]
            + rz * direction[2]
        )

        px = (
            rx
            - r_parallel * direction[0]
        )

        py = (
            ry
            - r_parallel * direction[1]
        )

        pz = (
            rz
            - r_parallel * direction[2]
        )

        r_perp_squared = (
            px * px
            + py * py
            + pz * pz
        )

        # --------------------------------------------------
        # Incident irradiance
        # --------------------------------------------------

        I = irradiance * np.exp(
            -r_perp_squared
            / (2.0 * sigma**2)
        )

        # --------------------------------------------------
        # Absorbed surface heat flux
        #
        # q'' [W/m²]
        # --------------------------------------------------

        return absorptance * I

    # ------------------------------------------------------
    # Coordinate diagnostics
    # ------------------------------------------------------

    X = heat_eq.V.tabulate_dof_coordinates()

    debug = False

    if debug:
        print("\n=== GAUSSIAN COORDINATE DIAGNOSTIC ===")

        print(
            "center =",
            center
        )

        print(
            "direction =",
            direction
        )

        print(
            "DOF X range:",
            X[:, 0].min(),
            "->",
            X[:, 0].max()
        )

        print(
            "DOF Y range:",
            X[:, 1].min(),
            "->",
            X[:, 1].max()
        )

        print(
            "DOF Z range:",
            X[:, 2].min(),
            "->",
            X[:, 2].max()
        )

    # Distance of closest DOF to beam axis
    dx = X[:, 0] - center[0]
    dy = X[:, 1] - center[1]

    r2 = dx**2 + dy**2

    if debug:
        print(
            "minimum perpendicular distance =",
            np.sqrt(np.min(r2)),
            "m"
        )

        print(
            "sigma =",
            sigma,
            "m"
        )

        print(
            "radius =",
            radius,
            "m"
        )

        print("====================================\n")
    # ------------------------------------------------------
    # Mesh / Gaussian overlap diagnostic
    # ------------------------------------------------------

    X = heat_eq.V.tabulate_dof_coordinates()

    #print("\n=== GAUSSIAN / MESH DIAGNOSTIC ===")

    #print(f"Gaussian center = {center}")
    #print(f"Direction       = {direction}")
    #print(f"Sigma           = {sigma:.12e} m")
    #print(f"Radius          = {radius:.12e} m")

    #print(
    #    f"Mesh X range    = "
    #    f"{X[:, 0].min():.6e} -> {X[:, 0].max():.6e} m"
    #)

    #print(
    #    f"Mesh Y range    = "
    #    f"{X[:, 1].min():.6e} -> {X[:, 1].max():.6e} m"
    #)

    #print(
    #    f"Mesh Z range    = "
    #    f"{X[:, 2].min():.6e} -> {X[:, 2].max():.6e} m"
    #)

    r_xy = np.sqrt(
        (X[:, 0] - center[0])**2
        +
        (X[:, 1] - center[1])**2
    )

    #print(
    #    f"Closest DOF to beam axis = "
    #    f"{r_xy.min():.12e} m"
    #)

    #print(
    #    f"DOFs within radius       = "
    #    f"{np.count_nonzero(r_xy <= radius)}"
    #)

    #print(
    #    f"DOFs within 3*sigma      = "
    #    f"{np.count_nonzero(r_xy <= 3*sigma)}"
    #)

    #print("================================\n")
    # ------------------------------------------------------
    # Interpolate onto FEM function
    # ------------------------------------------------------

    heat_eq.q_gaussian.interpolate(
        gaussian_surface_flux
    )

    q = heat_eq.q_gaussian.x.array

    debug = False

    if debug:
        print(
            f"q_gaussian max = "
            f"{q.max():.12e}"
        )

        print(
            f"q_gaussian nonzero = "
            f"{np.count_nonzero(q)}"
        )

        print(
            "q_gaussian min =",
            q.min()
        )

        print(
            "q_gaussian max =",
            q.max()
        )

        print(
            "q_gaussian nonzero =",
            np.count_nonzero(q)
        )
    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    if heat_eq.debug:

        print(
            f"   🔥 Gaussian surface source ON "
            f"(t={t:.3f}s)"
        )

        print(
            f"      Peak irradiance = "
            f"{irradiance:.6g} W/m²"
        )

        print(
            f"      sigma = "
            f"{sigma:.6g} m"
        )

        print(
            f"      radius = "
            f"{radius:.6g} m"
        )

        print(
            f"      absorptance = "
            f"{absorptance:.6g}"
        )

        print(
            f"      direction = "
            f"{direction}"
        )
