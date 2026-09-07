# orchestrator.py

import os
import sys
import yaml

from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem

from modules.physics.fields import build_fields
from modules.physics.heat_equation import HeatEquation
from modules.physics.solver import HeatSolver
import numpy as np
from modules.visualization.render import HeatVisualizer
from modules.visualization.create_gif import create_gif_stream


def apply_initial_condition(u_n, mesh, config):
    ic_cfg = config.get("simulation", {}).get("initial_condition", {})
    coords = mesh.geometry.x

    if ic_cfg.get("type") != "hotspot":
        raise ValueError(" Only hotspot IC supported for now")

    # -------------------------
    # Parameters
    # -------------------------
    ambient = float(ic_cfg.get("ambient", 298.15))
    delta = float(ic_cfg.get("delta", 50.0))

    center = np.array(ic_cfg.get("center", coords.mean(axis=0)))
    radius = float(ic_cfg.get("radius", 0.02))

    # -------------------------
    # Base field (ambient)
    # -------------------------
    u_n.x.array[:] = ambient

    # -------------------------
    # Hotspot
    # -------------------------
    dist = np.linalg.norm(coords - center, axis=1)
    mask = dist < radius

    num_hot = np.sum(mask)

    if num_hot == 0:
        raise ValueError(" No nodes inside hotspot radius")

    u_n.x.array[mask] += delta

    # -------------------------
    # Diagnostics
    # -------------------------
    print("\n🔥 Initial condition applied:")
    print(f"   ambient = {ambient} K")
    print(f"   delta   = {delta} K")
    print(f"   radius  = {radius}")
    print(f"   hot nodes = {num_hot}")
    print(f"   max(u0) = {u_n.x.array.max():.2f} K")
    print(f"   min(u0) = {u_n.x.array.min():.2f} K")

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f" Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "surface" not in config:
        raise ValueError(" Missing 'surface' section")

    return config


def load_mesh(xdmf_path: str):
    if not os.path.exists(xdmf_path):
        raise FileNotFoundError(f" Mesh file not found: {xdmf_path}")

    print(f"📥 Loading mesh: {xdmf_path}")

    with XDMFFile(MPI.COMM_WORLD, xdmf_path, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    return mesh


def run_simulation(sim_name: str):
    print(f"🚀 Running simulation: {sim_name}")

    base_input = os.path.join("inputs", sim_name)
    base_output = os.path.join("outputs", sim_name)

    config_path = os.path.join(base_input, "parameters", "parameters1.yaml")

    # 1. Load config
    config = load_config(config_path)

    # 2. Load mesh
    mesh_path = os.path.join(base_output, "mesh", "mesh.xdmf")
    mesh = load_mesh(mesh_path)

    visualizer = HeatVisualizer(mesh, output_dir=f"outputs/{sim_name}/frames")
    # 3. Validate mesh
    topo_dim = mesh.topology.dim
    geo_dim = mesh.geometry.dim

    print("\n✅ Mesh loaded successfully")
    print(f"   - Topology dimension: {topo_dim}")
    print(f"   - Geometry dimension: {geo_dim}")

    if topo_dim != 2 or geo_dim != 3:
        raise ValueError(" Expected surface mesh (2D in 3D)")

    # 4. Function space
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # 5. Initial condition
    u_n = fem.Function(V)
    apply_initial_condition(u_n, mesh, config)


    # 6. Build physical fields
    fields = build_fields(mesh, config)

    # 7. Time step (from config or default)
    sim_cfg = config.get("simulation", {})
    time_cfg = sim_cfg.get("time", {})

    dt = time_cfg.get("dt", 0.01)
    T = time_cfg.get("T", 1.0)

    print(f"Initial max temperature: {u_n.x.array.max():.2f}")
    # 8. Heat equation
    heat_eq = HeatEquation(mesh, V, fields, dt)

    # 9. Solver
    solver = HeatSolver(V, heat_eq, u_n)

    # 10. Callback (monitor)
    def callback(step, t, u):
        if step % 5 == 0:   # save every 5 steps
            print(f"📸 Saving frame {step}")
            visualizer.save_frame(u, step)

    # 11. Run simulation
    solver.run(T=T, callback=callback, save_every=1)

    print("✅ Simulation complete")

    visualizer.create_gif(f"outputs/{sim_name}/heat.gif", fps=10)

    frames_dir = f"outputs/{sim_name}/frames"
    gif_path = f"outputs/{sim_name}/heat.gif"

    create_gif_from_frames(frames_dir, gif_path, fps=10)

    print("✅ Visualization complete")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <simulation_name>")
        sys.exit(1)

    sim_name = sys.argv[1]

    try:
        run_simulation(sim_name)
    except Exception as e:
        print(f" Failed: {e}")
        sys.exit(1)
