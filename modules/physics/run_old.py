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

    Pipeline:

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

    Physics options:

        fixed_heat
            Volumetric heat generation [W] or [J/step]
            inside a rectangular prism.

        fixed_temperature
            Dirichlet temperature constraint [K]
            inside a rectangular prism.

        convection
            Surface heat loss.

        radiation
            Surface radiative heat loss.
    """

    print("\n🚀 Building simulation...")

    # ==========================================================
    # FUNCTION SPACE
    # ==========================================================

    V = fem.functionspace(
        mesh,
        ("CG", 1)
    )

    print("   ✔ Function space created")

    # ==========================================================
    # PHYSICAL FIELDS
    # ==========================================================

    fields = build_fields(
        mesh,
        config
    )

    print("   ✔ Physical fields created")

    # ==========================================================
    # PHYSICS CONFIGURATION
    # ==========================================================

    physics = (
        config["simulation"]
        .get("physics", {})
    )

    fixed_heat = physics.get(
        "fixed_heat",
        {}
    )

    fixed_temperature = physics.get(
        "fixed_temperature",
        {}
    )

    convection = physics.get(
        "convection",
        {}
    )

    radiation = physics.get(
        "radiation",
        {}
    )

    print("\n=== PHYSICS CONFIG ===")

    print(
        f"   Fixed heat        : "
        f"{'ON' if fixed_heat.get('enabled', False) else 'OFF'}"
    )

    print(
        f"   Fixed temperature : "
        f"{'ON' if fixed_temperature.get('enabled', False) else 'OFF'}"
    )

    print(
        f"   Convection        : "
        f"{'ON' if convection.get('enabled', False) else 'OFF'}"
    )

    print(
        f"   Radiation         : "
        f"{'ON' if radiation.get('enabled', False) else 'OFF'}"
    )

    # ==========================================================
    # FIXED HEAT DIAGNOSTICS
    # ==========================================================

    if fixed_heat.get("enabled", False):

        region = fixed_heat.get(
            "region",
            {}
        )

        print("\n   🔥 Fixed heat")

        if "power" in fixed_heat:

            print(
                f"      Power       = "
                f"{float(fixed_heat['power']):.6e} W"
            )

        elif "energy_per_step" in fixed_heat:

            print(
                f"      Energy/step = "
                f"{float(fixed_heat['energy_per_step']):.6e} J"
            )

        else:

            raise ValueError(
                "fixed_heat must define either "
                "'power' or 'energy_per_step'."
            )

        if region.get("type", "").lower() != "box":

            raise ValueError(
                "fixed_heat.region.type must be 'box'."
            )

        print(
            f"      Center      = "
            f"{region['center']}"
        )

        print(
            f"      Length X    = "
            f"{float(region['length_x']):.6e} m"
        )

        print(
            f"      Length Y    = "
            f"{float(region['length_y']):.6e} m"
        )

        print(
            f"      Length Z    = "
            f"{float(region['length_z']):.6e} m"
        )

        print(
            f"      t_start     = "
            f"{float(fixed_heat.get('t_start', 0.0)):.6g} s"
        )

        print(
            f"      t_end       = "
            f"{float(fixed_heat.get('t_end', np.inf)):.6g} s"
        )

    # ==========================================================
    # FIXED TEMPERATURE DIAGNOSTICS
    # ==========================================================

    if fixed_temperature.get("enabled", False):

        region = fixed_temperature.get(
            "region",
            {}
        )

        print("\n   🌡 Fixed temperature")

        temperature = float(
            fixed_temperature["temperature"]
        )

        print(
            f"      Temperature = "
            f"{temperature:.6f} K"
        )

        if region.get("type", "").lower() != "box":

            raise ValueError(
                "fixed_temperature.region.type must be 'box'."
            )

        print(
            f"      Center      = "
            f"{region['center']}"
        )

        print(
            f"      Length X    = "
            f"{float(region['length_x']):.6e} m"
        )

        print(
            f"      Length Y    = "
            f"{float(region['length_y']):.6e} m"
        )

        print(
            f"      Length Z    = "
            f"{float(region['length_z']):.6e} m"
        )

    # ==========================================================
    # TIME PARAMETERS
    # ==========================================================

    time_config = config["simulation"]["time"]

    dt = float(
        time_config["dt"]
    )

    T = float(
        time_config["T"]
    )

    save_every = (
        config["simulation"]
        .get("output", {})
        .get("save_every", 10)
    )

    print("\n=== TIME CONFIG ===")

    print(
        f"   dt         = {dt} s"
    )

    print(
        f"   T          = {T} s"
    )

    print(
        f"   save_every = {save_every}"
    )

    if dt <= 0.0:
        raise ValueError(
            "simulation.time.dt must be > 0."
        )

    if T <= 0.0:
        raise ValueError(
            "simulation.time.T must be > 0."
        )

    # ==========================================================
    # INITIAL CONDITION
    # ==========================================================

    ic = config["simulation"].get(
        "initial_condition",
        {}
    )

    ambient = float(
        ic["ambient"]
    )

    u_n = fem.Function(V)

    ic_type = ic.get(
        "type",
        "uniform"
    ).lower()

    # ----------------------------------------------------------
    # Uniform initial condition
    # ----------------------------------------------------------

    if ic_type == "uniform":

        u_n.x.array[:] = ambient

        print(
            f"\n   ✔ Uniform IC = "
            f"{ambient} K"
        )

    # ----------------------------------------------------------
    # Hotspot initial condition
    # ----------------------------------------------------------

    elif ic_type == "hotspot":

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

        if radius <= 0.0:

            raise ValueError(
                "initial_condition.radius must be > 0."
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

        u_n.interpolate(
            hotspot
        )

        print(
            "\n   ✔ Hotspot IC"
        )

        print(
            f"      ambient = "
            f"{ambient} K"
        )

        print(
            f"      delta   = "
            f"{delta} K"
        )

        print(
            f"      center  = "
            f"{center.tolist()}"
        )

        print(
            f"      radius  = "
            f"{radius} m"
        )

    else:

        raise ValueError(
            f"Unknown initial condition type "
            f"'{ic_type}'. "
            f"Expected 'uniform' or 'hotspot'."
        )

    # ==========================================================
    # INITIAL TEMPERATURE DIAGNOSTICS
    # ==========================================================

    Tmin = np.min(
        u_n.x.array
    )

    Tmax = np.max(
        u_n.x.array
    )

    print(
        "\n   ✔ Initial temperature range:"
    )

    print(
        f"      Tmin = "
        f"{Tmin:.3f} K"
    )

    print(
        f"      Tmax = "
        f"{Tmax:.3f} K"
    )

    # ==========================================================
    # HEAT EQUATION
    # ==========================================================

    heat_eq = HeatEquation(
        mesh=mesh,
        V=V,
        fields=fields,
        dt=dt,
        config=config,
        debug=False
    )

    print(
        "   ✔ Heat equation initialized"
    )

    # ==========================================================
    # FIXED TEMPERATURE CHECK
    # ==========================================================

    if fixed_temperature.get("enabled", False):

        if not hasattr(
            heat_eq,
            "fixed_temperature_bcs"
        ):

            raise RuntimeError(
                "fixed_temperature is enabled, "
                "but HeatEquation did not create "
                "fixed_temperature_bcs."
            )

        print(
            f"   ✔ Fixed-temperature BCs: "
            f"{len(heat_eq.fixed_temperature_bcs)}"
        )

    # ==========================================================
    # FIXED HEAT CHECK
    # ==========================================================

    if fixed_heat.get("enabled", False):

        if not hasattr(
            heat_eq,
            "q_fixed_heat"
        ):

            raise RuntimeError(
                "fixed_heat is enabled, "
                "but HeatEquation did not initialize "
                "q_fixed_heat."
            )

        if not hasattr(
            heat_eq,
            "fixed_heat_volume"
        ):

            raise RuntimeError(
                "fixed_heat is enabled, "
                "but HeatEquation did not calculate "
                "the fixed-heat volume."
            )

        print(
            f"   ✔ Fixed-heat volume: "
            f"{heat_eq.fixed_heat_volume:.6e} m³"
        )

    # ==========================================================
    # ENERGY BALANCE
    # ==========================================================

    heat_eq.energy_balance.initialize(
        u_n
    )

    print(
        "   ✔ Energy balance initialized"
    )

    # ==========================================================
    # SOLVER
    # ==========================================================

    solver = HeatSolver(
        V=V,
        heat_equation=heat_eq,
        u_n=u_n,
        sim_name=sim_name,
        output_dir=output_dir,
        roi_config=config.get(
            "roi",
            {}
        ),
    )

    # ==========================================================
    # RUN
    # ==========================================================

    solution = solver.run(
        T,
        save_every=save_every
    )

    return solution
