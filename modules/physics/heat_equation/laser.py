import numpy as np

from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary
from ufl import Measure


# ============================================================
# INITIALIZE LASER
# ============================================================

def initialize_laser(heat_eq, source):
    """
    Initialize a laser as a Gaussian incident surface source.

    The laser is defined by:

        power       : total optical power [W]
        center      : point on the beam axis [m]
        direction   : propagation direction
        radius      : beam radius, interpreted as 3*sigma [m]
        absorptance : fraction of incident radiation absorbed [-]

    The Gaussian beam is normalized so that:

        integral(I dA) = power

    over an infinite plane perpendicular to the beam axis.

    The actual absorbed heat flux on a surface is:

        q_abs = absorptance * I(r) * cos(theta)

    where:

        cos(theta) = max(0, n · (-direction))

    """

    mesh = heat_eq.mesh

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # --------------------------------------------------
    # Direction
    # --------------------------------------------------

    direction = np.asarray(
        source["direction"],
        dtype=float,
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
    # We identify all external facets.
    # The Gaussian flux itself determines where the
    # laser deposits significant energy.
    # --------------------------------------------------

    boundary_facets = locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.ones(
            x.shape[1],
            dtype=bool,
        ),
    )

    if len(boundary_facets) == 0:
        raise RuntimeError(
            " Laser source: no boundary facets found"
        )

    # --------------------------------------------------
    # Mark laser facets
    # --------------------------------------------------

    from dolfinx.mesh import meshtags

    values = np.ones(
        len(boundary_facets),
        dtype=np.int32,
    )

    heat_eq.laser_facets = boundary_facets

    heat_eq.laser_facet_tags = meshtags(
        mesh,
        fdim,
        boundary_facets,
        values,
    )

    heat_eq.ds_laser = Measure(
        "ds",
        domain=mesh,
        subdomain_data=heat_eq.laser_facet_tags,
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

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    if heat_eq.debug:

        print("\n🔬 Laser source initialized")

        print(
            f"   direction = {direction}"
        )

        print(
            f"   boundary facets = "
            f"{len(boundary_facets)}"
        )


# ============================================================
# UPDATE LASER SOURCE
# ============================================================

def update_laser_source(
    heat_eq,
    source,
    t,
):
    """
    Update the laser surface heat flux.

    The laser uses:

        power       [W]
        center      [m]
        direction   [-]
        radius      [m]
        absorptance [-]

    The Gaussian beam has:

        sigma = radius / 3

    and:

        I0 = P / (2*pi*sigma^2)

    The absorbed surface heat flux is:

        q_abs =
            absorptance
            * I(r)
            * max(0, n · (-direction))

    """

    # --------------------------------------------------
    # Time interval
    # --------------------------------------------------

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
    # Beam center
    # --------------------------------------------------

    center = np.asarray(
        source["center"],
        dtype=float,
    )

    if center.size != 3:

        raise ValueError(
            "physics.source.center must contain "
            "three coordinates"
        )

    # --------------------------------------------------
    # Propagation direction
    # --------------------------------------------------

    direction = np.asarray(
        source["direction"],
        dtype=float,
    )

    if direction.size != 3:

        raise ValueError(
            "physics.source.direction must contain "
            "three components"
        )

    norm = np.linalg.norm(
        direction
    )

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
            1.0,
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
    # The integral over an infinite plane
    # perpendicular to the beam is:
    #
    # P = integral(I dA)
    #
    # Therefore:
    #
    # I0 = P / (2*pi*sigma^2)
    # --------------------------------------------------

    intensity_peak = (
        power
        / (
            2.0
            * np.pi
            * sigma**2
        )
    )

    # --------------------------------------------------
    # Geometry information
    # --------------------------------------------------

    # We need the mesh coordinates to calculate
    # the local Gaussian intensity.
    #
    # x contains the coordinates of the FEM
    # interpolation points.
    # --------------------------------------------------

    def gaussian_surface_flux(x):

        # --------------------------------------------------
        # Vector from beam center to evaluation point
        # --------------------------------------------------

        rx = x[0] - center[0]
        ry = x[1] - center[1]
        rz = x[2] - center[2]

        # --------------------------------------------------
        # Distance parallel to beam axis
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
        # Gaussian irradiance
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
        # At this stage this is the incident
        # irradiance on a plane perpendicular
        # to the beam.
        #
        # The surface-normal correction cannot be
        # calculated here because x does not contain
        # the local FEM facet normal.
        #
        # Therefore the Function stores the Gaussian
        # incident irradiance.
        # --------------------------------------------------

        return intensity

    # --------------------------------------------------
    # Interpolate Gaussian irradiance
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
            f"      center = "
            f"{center}"
        )

        print(
            f"      direction = "
            f"{direction}"
        )
