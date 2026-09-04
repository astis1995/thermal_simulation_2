"""
Fixed volumetric heat source.

Adds uniform heat generation inside a selected volume.

Two input modes are supported:

    power [W]
        Total heat-generation rate deposited in the selected volume.

    energy_per_step [J]
        Energy deposited during every timestep.

The solver converts this into a volumetric heat-generation rate
q_vol [W/m^3] and adds it to the transient heat equation.
"""

import numpy as np
from dolfinx import fem
from dolfinx.mesh import locate_entities, meshtags
from ufl import Measure
from mpi4py import MPI
from dolfinx import mesh

def _inside_function(region):

    kind = region.get("type", "box").lower()

    if kind != "box":
        raise ValueError(
            f"Unsupported fixed_heat region type '{kind}'. "
            "Use 'box'."
        )

    center = np.asarray(
        region["center"],
        dtype=float,
    )

    lengths = np.asarray(
        [
            region["length_x"],
            region["length_y"],
            region["length_z"],
        ],
        dtype=float,
    )

    if center.size != 3:
        raise ValueError(
            "fixed_heat box requires center [x, y, z]."
        )

    if np.any(lengths <= 0.0):
        raise ValueError(
            "fixed_heat box requires "
            "length_x, length_y and length_z > 0."
        )

    half = 0.5 * lengths

    xmin = center[0] - half[0]
    xmax = center[0] + half[0]

    ymin = center[1] - half[1]
    ymax = center[1] + half[1]

    zmin = center[2] - half[2]
    zmax = center[2] + half[2]

    def inside(x):

        return (
            (x[0] >= xmin)
            & (x[0] <= xmax)
            & (x[1] >= ymin)
            & (x[1] <= ymax)
            & (x[2] >= zmin)
            & (x[2] <= zmax)
        )

    return inside

def initialize_fixed_heat(heat_eq, config):
    """
    Locate cells whose centroids are inside the selected box and
    create the UFL integration measure used by HeatEquation.
    """

    heat_eq.q_fixed_heat = fem.Constant(
        heat_eq.mesh,
        0.0,
    )

    heat_eq.dx_fixed_heat = None
    heat_eq.fixed_heat_volume = 0.0
    heat_eq.fixed_heat_mode = None
    heat_eq.fixed_heat_input = 0.0
    heat_eq.fixed_heat_cells = np.array([], dtype=np.int32)

    if not config or not config.get("enabled", False):
        return

    if heat_eq.mesh.topology.dim != 3:
        raise ValueError(
            "fixed_heat currently requires a 3D volume mesh."
        )

    region = config["region"]

    if region.get("type", "box").lower() != "box":
        raise ValueError(
            "fixed_heat region type must be 'box'."
        )

    # ---------------------------------------------------------
    # Box definition
    # ---------------------------------------------------------

    center = np.asarray(
        region["center"],
        dtype=float,
    )

    lengths = np.asarray(
        [
            region["length_x"],
            region["length_y"],
            region["length_z"],
        ],
        dtype=float,
    )

    if center.size != 3:
        raise ValueError(
            "fixed_heat box requires center [x, y, z]."
        )

    if lengths.size != 3 or np.any(lengths <= 0.0):
        raise ValueError(
            "fixed_heat box requires positive "
            "length_x, length_y and length_z."
        )

    half = 0.5 * lengths

    xmin = center[0] - half[0]
    xmax = center[0] + half[0]

    ymin = center[1] - half[1]
    ymax = center[1] + half[1]

    zmin = center[2] - half[2]
    zmax = center[2] + half[2]

    # ---------------------------------------------------------
    # Find cell centroids
    # ---------------------------------------------------------

    tdim = heat_eq.mesh.topology.dim

    num_cells = heat_eq.mesh.topology.index_map(tdim).size_local

    cells_all = np.arange(
        num_cells,
        dtype=np.int32,
    )

    cell_centers = mesh.compute_midpoints(
        heat_eq.mesh,
        tdim,
        cells_all,
    )

    print("\n=== CELL CENTROID DEBUG ===")

    print(
        f"Number of local cells: {len(cells_all)}"
    )

    print(
        f"Centroid X: "
        f"[{cell_centers[:,0].min():.6e}, "
        f"{cell_centers[:,0].max():.6e}]"
    )

    print(
        f"Centroid Y: "
        f"[{cell_centers[:,1].min():.6e}, "
        f"{cell_centers[:,1].max():.6e}]"
    )

    print(
        f"Centroid Z: "
        f"[{cell_centers[:,2].min():.6e}, "
        f"{cell_centers[:,2].max():.6e}]"
    )

    print("\nFirst 10 cell centroids:")

    for i in range(min(10, len(cell_centers))):
        print(
            f"   {i}: "
            f"[{cell_centers[i,0]:.6e}, "
            f"{cell_centers[i,1]:.6e}, "
            f"{cell_centers[i,2]:.6e}]"
        )

    print("\nHeat box:")

    print(
        f"   X: [{xmin:.6e}, {xmax:.6e}]"
    )

    print(
        f"   Y: [{ymin:.6e}, {ymax:.6e}]"
    )

    print(
        f"   Z: [{zmin:.6e}, {zmax:.6e}]"
    )

    # Count how many centroids satisfy each coordinate independently

    x_inside = (
        (cell_centers[:,0] >= xmin)
        & (cell_centers[:,0] <= xmax)
    )

    y_inside = (
        (cell_centers[:,1] >= ymin)
        & (cell_centers[:,1] <= ymax)
    )

    z_inside = (
        (cell_centers[:,2] >= zmin)
        & (cell_centers[:,2] <= zmax)
    )

    print("\nIndependent selection:")

    print(
        f"   X inside: {np.count_nonzero(x_inside)}"
    )

    print(
        f"   Y inside: {np.count_nonzero(y_inside)}"
    )

    print(
        f"   Z inside: {np.count_nonzero(z_inside)}"
    )

    xyz_inside = (
        x_inside
        & y_inside
        & z_inside
    )

    print(
        f"   XYZ inside: "
        f"{np.count_nonzero(xyz_inside)}"
    )

    print("============================\n")

    cells = cells_all[xyz_inside]

    # ---------------------------------------------------------
    # Select cells whose centroid is inside the box
    # ---------------------------------------------------------

    inside = (
        (cell_centers[:, 0] >= xmin)
        & (cell_centers[:, 0] <= xmax)
        & (cell_centers[:, 1] >= ymin)
        & (cell_centers[:, 1] <= ymax)
        & (cell_centers[:, 2] >= zmin)
        & (cell_centers[:, 2] <= zmax)
    )

    cells = cells_all[inside]

    if len(cells) == 0:
        raise RuntimeError(
            "Fixed-heat region contains no cell centroids. "
            "Check the box coordinates and mesh."
        )

    cells = np.asarray(
        cells,
        dtype=np.int32,
    )

    cells.sort()

    heat_eq.fixed_heat_cells = cells

    # ---------------------------------------------------------
    # Tag selected cells
    # ---------------------------------------------------------

    values = np.ones(
        len(cells),
        dtype=np.int32,
    )

    heat_eq.fixed_heat_tags = meshtags(
        heat_eq.mesh,
        tdim,
        cells,
        values,
    )

    heat_eq.dx_fixed_heat = Measure(
        "dx",
        domain=heat_eq.mesh,
        subdomain_data=heat_eq.fixed_heat_tags,
    )

    # ---------------------------------------------------------
    # Calculate selected volume
    # ---------------------------------------------------------

    one = fem.Constant(
        heat_eq.mesh,
        1.0,
    )

    local_volume = fem.assemble_scalar(
        fem.form(
            one * heat_eq.dx_fixed_heat(1)
        )
    )

    heat_eq.fixed_heat_volume = heat_eq.mesh.comm.allreduce(
        local_volume,
        op=MPI.SUM,
    )

    if heat_eq.fixed_heat_volume <= 0.0:
        raise RuntimeError(
            "Fixed-heat region has zero volume."
        )

    # ---------------------------------------------------------
    # Determine input mode
    # ---------------------------------------------------------

    has_power = "power" in config
    has_energy = "energy_per_step" in config

    if has_power == has_energy:
        raise ValueError(
            "fixed_heat must define exactly one of "
            "'power' [W] or 'energy_per_step' [J]."
        )

    if has_power:

        value = float(
            config["power"]
        )

        if value < 0.0:
            raise ValueError(
                "fixed_heat.power must be >= 0."
            )

        heat_eq.fixed_heat_mode = "power"

    else:

        value = float(
            config["energy_per_step"]
        )

        if value < 0.0:
            raise ValueError(
                "fixed_heat.energy_per_step must be >= 0."
            )

        heat_eq.fixed_heat_mode = "energy_per_step"

    heat_eq.fixed_heat_input = value

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    if heat_eq.debug:

        print("\n🔥 Fixed volumetric heat region")

        print(
            f"   center      = "
            f"[{center[0]:.6e}, "
            f"{center[1]:.6e}, "
            f"{center[2]:.6e}] m"
        )

        print(
            f"   lengths     = "
            f"[{lengths[0]:.6e}, "
            f"{lengths[1]:.6e}, "
            f"{lengths[2]:.6e}] m"
        )

        print(
            f"   X range     = "
            f"[{xmin:.6e}, {xmax:.6e}] m"
        )

        print(
            f"   Y range     = "
            f"[{ymin:.6e}, {ymax:.6e}] m"
        )

        print(
            f"   Z range     = "
            f"[{zmin:.6e}, {zmax:.6e}] m"
        )

        print(
            f"   cells       = {len(cells)}"
        )

        print(
            f"   volume      = "
            f"{heat_eq.fixed_heat_volume:.6e} m³"
        )

        print(
            f"   mode        = "
            f"{heat_eq.fixed_heat_mode}"
        )

        print(
            f"   input       = "
            f"{value:.6e}"
        )

def update_fixed_heat(heat_eq, config, t):
    """
    Update q_fixed_heat [W/m^3].

    For power mode:
        total power = power

    For energy_per_step mode:
        total power = energy_per_step / dt
    """
    if not config or not config.get("enabled", False):
        heat_eq.q_fixed_heat.value = 0.0
        return

    t0 = float(config.get("t_start", 0.0))
    t1 = float(config.get("t_end", np.inf))

    if not (t0 <= t <= t1):
        heat_eq.q_fixed_heat.value = 0.0
        return

    if heat_eq.fixed_heat_mode == "power":
        total_power = heat_eq.fixed_heat_input
    elif heat_eq.fixed_heat_mode == "energy_per_step":
        total_power = heat_eq.fixed_heat_input / heat_eq.dt
    else:
        raise RuntimeError(
            "fixed_heat has not been initialized correctly."
        )

    heat_eq.q_fixed_heat.value = (
        total_power / heat_eq.fixed_heat_volume
    )

    if heat_eq.debug:
        energy = total_power * heat_eq.dt
        print(
            f"🔥 Fixed heat ON (t={t:.3f}s): "
            f"q_vol={heat_eq.q_fixed_heat.value:.6e} W/m³, "
            f"P={total_power:.6e} W, "
            f"E_step={energy:.6e} J"
        )
