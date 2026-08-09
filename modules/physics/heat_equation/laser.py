import numpy as np

from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary
from ufl import Measure


def initialize_laser(heat_eq, source):
    """
    Initialize the laser as an incident surface source.

    The laser is defined by:
        - source point
        - propagation direction
        - beam radius

    The actual power is supplied by the source configuration
    (and can later be supplied dynamically by laser.py).
    """

    mesh = heat_eq.mesh

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # --------------------------------------------------
    # Direction
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Boundary facets
    #
    # For now we identify all external facets.
    # Actual first-hit filtering will be added here.
    # --------------------------------------------------

    boundary_facets = locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.ones(
            x.shape[1],
            dtype=bool
        )
    )

    if len(boundary_facets) == 0:
        raise RuntimeError(
            "❌ Laser source: no boundary facets found"
        )

    # --------------------------------------------------
    # Mark laser facets
    # --------------------------------------------------

    from dolfinx.mesh import meshtags

    values = np.ones(
        len(boundary_facets),
        dtype=np.int32
    )

    heat_eq.laser_facets = boundary_facets

    heat_eq.laser_facet_tags = meshtags(
        mesh,
        fdim,
        boundary_facets,
        values
    )

    heat_eq.ds_laser = Measure(
        "ds",
        domain=mesh,
        subdomain_data=heat_eq.laser_facet_tags
    )

    # --------------------------------------------------
    # Surface heat-flux field
    # --------------------------------------------------

    heat_eq.q_laser = fem.Function(
        heat_eq.V
    )

    heat_eq.q_laser.name = (
        "LaserSurfaceFlux"
    )

    heat_eq.q_laser.x.array[:] = 0.0

    if heat_eq.debug:

        print("\n🔬 Laser source initialized")

        print(
            f"   direction = {direction}"
        )

        print(
            f"   boundary facets = "
            f"{len(boundary_facets)}"
        )


def update_laser_source(
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

    # --------------------------------------------------
    # Source OFF
    # --------------------------------------------------

    if not (t0 <= t <= t1):

        heat_eq.q_laser.x.array[:] = 0.0

        if heat_eq.debug:

            print(
                f"   ❄ Laser OFF "
                f"(t={t:.3f}s)"
            )

        return

    # --------------------------------------------------
    # Total laser power
    # --------------------------------------------------

    power = float(
        source["power"]
    )

    if power <= 0.0:

        raise ValueError(
            "physics.source.power must be > 0"
        )

    # --------------------------------------------------
    # Source point
    # --------------------------------------------------

    center = np.asarray(
        source["center"],
        dtype=float
    )

    # --------------------------------------------------
    # Propagation direction
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Beam radius
    #
    # radius = 3 sigma
    # --------------------------------------------------

    radius = float(
        source["radius"]
    )

    if radius <= 0.0:

        raise ValueError(
            "physics.source.radius "
            "must be > 0"
        )

    sigma = radius / 3.0

    # --------------------------------------------------
    # Absorptance
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Gaussian normalization
    #
    # P = integral(I dA)
    #
    # I0 = P / (2*pi*sigma²)
    # --------------------------------------------------

    intensity_peak = (
        power /
        (
            2.0
            * np.pi
            * sigma**2
        )
    )

    # --------------------------------------------------
    # Gaussian surface irradiance
    # --------------------------------------------------

    def gaussian_surface_flux(x):

        rx = x[0] - center[0]
        ry = x[1] - center[1]
        rz = x[2] - center[2]

        # --------------------------------------------------
        # Projection onto propagation direction
        # --------------------------------------------------

        r_parallel = (
            rx * direction[0]
            + ry * direction[1]
            + rz * direction[2]
        )

        # --------------------------------------------------
        # Perpendicular distance from beam axis
        # --------------------------------------------------

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
        # Incident irradiance [W/m²]
        # --------------------------------------------------

        intensity = (
            intensity_peak
            * np.exp(
                -r_perp_squared
                / (
                    2.0
                    * sigma**2
                )
            )
        )

        # --------------------------------------------------
        # Absorbed surface flux [W/m²]
        # --------------------------------------------------

        return absorptance * intensity

    # --------------------------------------------------
    # Apply to FEM surface field
    # --------------------------------------------------

    heat_eq.q_laser.interpolate(
        gaussian_surface_flux
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    if heat_eq.debug:

        print(
            f"   🔥 Laser ON "
            f"(t={t:.3f}s)"
        )

        print(
            f"      Power = "
            f"{power:.6g} W"
        )

        print(
            f"      Peak irradiance = "
            f"{intensity_peak:.6g} W/m²"
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
