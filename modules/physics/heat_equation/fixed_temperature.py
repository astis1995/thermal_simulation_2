"""
Fixed-temperature volume constraint.

Applies a Dirichlet temperature condition to every CG1 degree of
freedom located inside a user-defined geometric volume.

Supported regions:
    sphere:
        center: [x, y, z]  [m]
        radius: R            [m]

    box:
        min: [xmin, ymin, zmin] [m]
        max: [xmax, ymax, zmax] [m]

The constraint is static during the simulation. If the active
temperature constraint must change with time, the FEM matrix/boundary
conditions must be rebuilt when that change occurs.
"""

import numpy as np
from dolfinx import fem


def _inside_function(region):
    kind = region.get("type", "sphere").lower()

    if kind == "sphere":
        center = np.asarray(region["center"], dtype=float)
        radius = float(region["radius"])

        if center.size != 3 or radius <= 0.0:
            raise ValueError(
                "fixed_temperature sphere requires a 3D center "
                "and radius > 0."
            )

        def inside(x):
            r2 = (
                (x[0] - center[0]) ** 2
                + (x[1] - center[1]) ** 2
                + (x[2] - center[2]) ** 2
            )
            return r2 <= radius ** 2

        return inside

    if kind == "box":
        xmin = np.asarray(region["min"], dtype=float)
        xmax = np.asarray(region["max"], dtype=float)

        if xmin.size != 3 or xmax.size != 3 or np.any(xmax <= xmin):
            raise ValueError(
                "fixed_temperature box requires min/max 3D coordinates "
                "with max > min."
            )

        def inside(x):
            return (
                (x[0] >= xmin[0]) & (x[0] <= xmax[0])
                & (x[1] >= xmin[1]) & (x[1] <= xmax[1])
                & (x[2] >= xmin[2]) & (x[2] <= xmax[2])
            )

        return inside

    raise ValueError(
        f"Unsupported fixed_temperature region type '{kind}'. "
        "Use 'sphere' or 'box'."
    )


def initialize_fixed_temperature(heat_eq, config):
    """
    Create the Dirichlet BC for a temperature-fixed volume.
    """
    heat_eq.fixed_temperature_bcs = []
    heat_eq.fixed_temperature_value = None
    heat_eq.fixed_temperature_dofs = np.array([], dtype=np.int32)

    if not config or not config.get("enabled", False):
        return

    temperature = float(config["temperature"])
    region = config["region"]

    inside = _inside_function(region)

    dofs = fem.locate_dofs_geometrical(
        heat_eq.V,
        inside,
    )

    if len(dofs) == 0:
        raise RuntimeError(
            "Fixed-temperature region contains no CG1 degrees of freedom. "
            "Increase the region size or check its coordinates."
        )

    T_fixed = fem.Constant(
        heat_eq.mesh,
        temperature,
    )

    bc = fem.dirichletbc(
        T_fixed,
        dofs,
    )

    heat_eq.fixed_temperature_bcs = [bc]
    heat_eq.fixed_temperature_value = T_fixed
    heat_eq.fixed_temperature_dofs = dofs

    if heat_eq.debug:
        print("\n🌡 Fixed-temperature region")
        print(f"   temperature     = {temperature:.6f} K")
        print(f"   region type     = {region.get('type', 'sphere')}")
        print(f"   constrained DOFs = {len(dofs)}")
