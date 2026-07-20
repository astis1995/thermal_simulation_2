# modules/physics/run.py

import numpy as np

from dolfinx import fem

from .fields import build_fields
from .heat_equation import HeatEquation
from .solver import HeatSolver


def run_simulation(
    mesh,
    config,
    sim_name,
    output_dir
):
    """
    Complete thermal simulation pipeline.

    mesh
      ↓
    fields
      ↓
    function space
      ↓
    initial condition
      ↓
    heat equation
      ↓
    solver
      ↓
    run
    """

    print("\n🚀 Building simulation...")

    # --------------------------------------------------
    # Function space
    # --------------------------------------------------

    V = fem.functionspace(
        mesh,
        ("CG", 1)
    )

    print("   ✔ Function space created")

    # --------------------------------------------------
    # Physical fields
    # --------------------------------------------------

    fields = build_fields(mesh, config)

    # --------------------------------------------------
    # Time parameters
    # --------------------------------------------------

    dt = config["simulation"]["time"]["dt"]
    T = config["simulation"]["time"]["T"]

    save_every = (
        config["simulation"]
              .get("output", {})
              .get("save_every", 10)
    )
    # --------------------------------------------------
    # Initial condition
    # --------------------------------------------------

    ic = config["simulation"]["initial_condition"]

    ambient = ic["ambient"]

    u_n = fem.Function(V)

    ic_type = ic.get(
        "type",
        "uniform"
    ).lower()

    if ic_type == "hotspot":

        center = np.asarray(
            ic["center"],
            dtype=float
        )

        radius = float(
            ic["radius"]
        )

        delta = float(
            ic["delta"]
        )

        def hotspot(x):

            r2 = (
                (x[0] - center[0])**2
                + (x[1] - center[1])**2
                + (x[2] - center[2])**2
            )

            values = np.full(
                x.shape[1],
                ambient,
                dtype=np.float64
            )

            values[r2 <= radius**2] += delta

            return values

        u_n.interpolate(hotspot)

        print(
            f"   ✔ Hotspot IC"
        )
        print(
            f"      ambient = {ambient} K"
        )
        print(
            f"      delta   = {delta} K"
        )
        print(
            f"      center  = {center.tolist()}"
        )
        print(
            f"      radius  = {radius} m"
        )

    else:

        u_n.x.array[:] = ambient

        print(
            f"   ✔ Uniform IC = {ambient} K"
        )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    Tmin = np.min(u_n.x.array)
    Tmax = np.max(u_n.x.array)

    print(
        f"   ✔ Initial temperature range:"
    )
    print(
        f"      Tmin = {Tmin:.3f} K"
    )
    print(
        f"      Tmax = {Tmax:.3f} K"
    )

    # --------------------------------------------------
    # PDE
    # --------------------------------------------------

    heat_eq = HeatEquation(
        mesh=mesh,
        V=V,
        fields=fields,
        dt=dt,
        config=config,
        debug=True
    )

    # --------------------------------------------------
    # Solver
    # --------------------------------------------------

    solver = HeatSolver(
        V=V,
        heat_equation=heat_eq,
        u_n=u_n,
        sim_name=sim_name,
        output_dir=output_dir,
        roi_config=config.get("roi", {}),
    )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    solution = solver.run(
        T,
        save_every=save_every
    )

    return solution
