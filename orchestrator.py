# orchestrator.py

import os
import sys
import yaml

from mpi4py import MPI
from dolfinx.io import XDMFFile

from modules.stl_to_mesh import generate_mesh_from_stl
from modules.physics.run import run_simulation
from modules.stl_to_mesh.test_stl import validate_stl
from modules.stl_to_mesh.fix_stl import fix_stl

# ==========================================================
# CONFIG
# ==========================================================

def load_config(config_path: str):

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f" Config not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "domain" not in config:
        raise ValueError(
            " Missing 'domain' section"
        )

    return config


# ==========================================================
# MESH
# ==========================================================

def load_mesh(xdmf_path: str):

    if not os.path.exists(xdmf_path):
        raise FileNotFoundError(
            f" Mesh file not found: {xdmf_path}"
        )

    print(f"📥 Loading mesh: {xdmf_path}")

    with XDMFFile(
        MPI.COMM_WORLD,
        xdmf_path,
        "r"
    ) as xdmf:

        mesh = xdmf.read_mesh()

    return mesh


def ensure_mesh_exists(
    base_input: str,
    base_output: str,
    config: dict,
    parameters_file: str,
    cache: bool = True
):

    mesh_cfg = config["domain"]["mesh"]

    mesh_dir = os.path.join(
        base_output,
        mesh_cfg.get("output_dir", "mesh")
    )

    os.makedirs(
        mesh_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # Mesh filename derived from parameter file
    # --------------------------------------------------

    parameter_stem = os.path.splitext(
        os.path.basename(parameters_file)
    )[0]

    mesh_filename = (
        f"{parameter_stem}_mesh.xdmf"
    )

    mesh_path = os.path.join(
        mesh_dir,
        mesh_filename
    )

    # --------------------------------------------------
    # Cache handling
    # --------------------------------------------------

    if os.path.exists(mesh_path):

        if cache:

            print(
                f"📦 Using cached mesh: {mesh_path}"
            )

            return mesh_path

        print(
            "♻️ Rebuilding mesh (cache=False)"
        )

        try:
            os.remove(mesh_path)
        except OSError:
            pass

        # Remove matching HDF5 file
        h5_path = mesh_path.replace(
            ".xdmf",
            ".h5"
        )

        if os.path.exists(h5_path):

            try:
                os.remove(h5_path)
            except OSError:
                pass

    # --------------------------------------------------
    # STL
    # --------------------------------------------------

    stl_rel_path = config["domain"]["path"]

    stl_path = os.path.join(
        base_input,
        stl_rel_path
    )

    if not os.path.exists(stl_path):

        raise FileNotFoundError(
            f"❌ STL file not found: {stl_path}"
        )

    print(
        f"📥 STL source: {stl_path}"
    )

    # --------------------------------------------------
    # Validate original STL
    # --------------------------------------------------

    print("\n🔍 Validating STL...")

    stl_is_valid = validate_stl(
        stl_path=stl_path,
        config=config,
        verbose=True
    )

    # --------------------------------------------------
    # Repair STL if necessary
    # --------------------------------------------------

    if not stl_is_valid:

        print("\n⚠️ STL failed validation.")
        print("🔧 Attempting automatic repair...")

        fixed_stl_path = fix_stl(
            stl_path=stl_path,
            config=config,
            verbose=True
        )

        # --------------------------------------------------
        # Validate repaired STL
        # --------------------------------------------------

        print(
            "\n🔍 Validating repaired STL..."
        )

        fixed_is_valid = validate_stl(
            stl_path=fixed_stl_path,
            config=config,
            verbose=True
        )

        if not fixed_is_valid:

            raise RuntimeError(
                "\n❌ STL remains invalid after repair.\n"
                f"Original STL : {stl_path}\n"
                f"Repaired STL : {fixed_stl_path}\n"
                "\n"
                "Mesh generation with Gmsh was aborted."
            )

        print(
            "\n✅ Repaired STL passed validation."
        )

        # Use repaired STL from this point forward
        stl_path = str(
            fixed_stl_path
        )

    else:

        print(
            "\n✅ Original STL passed validation."
        )

    # --------------------------------------------------
    # Geometry representation
    # --------------------------------------------------

    geometry_representation = (
        config
        .get("simulation", {})
        .get("geometry", {})
        .get("representation", "surface")
    )

    print(
        f"\n📐 Geometry representation: "
        f"{geometry_representation}"
    )

    # --------------------------------------------------
    # Physics source configuration
    # --------------------------------------------------

    source_config = (
        config
        .get("simulation", {})
        .get("physics", {})
        .get("source")
    )

    # --------------------------------------------------
    # Generate mesh
    # --------------------------------------------------

    print("\n🔨 Generating mesh with Gmsh...")

    generate_mesh_from_stl(
        stl_path=stl_path,
        output_dir=mesh_dir,
        output_filename=mesh_filename,
        lc_min=mesh_cfg["lc_min"],
        lc_max=mesh_cfg["lc_max"],
        geometry_representation=geometry_representation,
        unit=config["domain"].get("unit", "mm"),
        source_config=source_config,
    )

    # --------------------------------------------------
    # Validate generated mesh
    # --------------------------------------------------

    if not os.path.exists(mesh_path):

        raise RuntimeError(
            f"❌ Mesh generation failed: "
            f"{mesh_path}"
        )

    h5_path = mesh_path.replace(
        ".xdmf",
        ".h5"
    )

    if not os.path.exists(h5_path):

        raise RuntimeError(
            f"❌ HDF5 mesh file missing: "
            f"{h5_path}"
        )

    print(
        f"\n✅ Mesh ready: {mesh_path}"
    )

    return mesh_path


def validate_mesh(mesh, config):

    topo_dim = mesh.topology.dim
    geo_dim = mesh.geometry.dim
    num_points = mesh.geometry.x.shape[0]

    representation = (
        config
        .get("simulation", {})
        .get("geometry", {})
        .get("representation", "surface")
        .lower()
    )

    expected_topology = (
        2 if representation == "surface" else 3
    )

    print("\n Mesh loaded successfully")
    print(f"   - Representation      : {representation}")
    print(f"   - Topology dimension  : {topo_dim}")
    print(f"   - Geometry dimension  : {geo_dim}")
    print(f"   - Number of points    : {num_points}")

    if topo_dim != expected_topology:
        raise ValueError(
            f" Expected topology={expected_topology}, "
            f"got {topo_dim}"
        )

    if geo_dim != 3:
        raise ValueError(
            f" Expected embedded 3D geometry, got {geo_dim}"
        )

    xyz_min = mesh.geometry.x.min(axis=0)
    xyz_max = mesh.geometry.x.max(axis=0)

    print("\n Bounding box [m]")

    print(
        f"   X: {xyz_min[0]:.6e} -> {xyz_max[0]:.6e}"
    )

    print(
        f"   Y: {xyz_min[1]:.6e} -> {xyz_max[1]:.6e}"
    )

    print(
        f"   Z: {xyz_min[2]:.6e} -> {xyz_max[2]:.6e}"
    )


# ==========================================================
# PIPELINE
# ==========================================================

def prepare_simulation(
    sim_name: str,
    parameters_file: str = "parameters1.yaml",
    cache: bool = True
):

    print(f"\n Preparing simulation: {sim_name}")
    print(f" Parameters file: {parameters_file}")
    print(f" Mesh cache enabled: {cache}")

    base_input = os.path.join(
        "inputs",
        sim_name
    )

    base_output = os.path.join(
        "outputs",
        sim_name
    )

    config_path = os.path.join(
        base_input,
        "parameters",
        parameters_file
    )

    config = load_config(config_path)

    mesh_path = ensure_mesh_exists(
        base_input=base_input,
        base_output=base_output,
        config=config,
        parameters_file=parameters_file,
        cache=cache
    )

    print(
        f"base_input  = "
        f"{os.path.abspath(base_input)}"
    )

    print(
        f"base_output = "
        f"{os.path.abspath(base_output)}"
    )

    print(
        f"mesh_path   = "
        f"{os.path.abspath(mesh_path)}"
    )

    mesh = load_mesh(mesh_path)

    validate_mesh(
        mesh,
        config
    )

    return (
        mesh,
        config,
        base_output
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python orchestrator.py "
            "<simulation_name>"
        )
        print(
            "  python orchestrator.py "
            "<simulation_name> "
            "<parameters.yaml>"
        )
        print(
            "  python orchestrator.py "
            "<simulation_name> "
            "<parameters.yaml> "
            "--rebuild"
        )

        sys.exit(1)

    sim_name = sys.argv[1]

    parameters_file = "parameters1.yaml"

    for arg in sys.argv[2:]:

        if (
            arg.endswith(".yaml")
            or arg.endswith(".yml")
        ):
            parameters_file = arg

    cache = "--rebuild" not in sys.argv

    try:

        mesh, config, base_output = prepare_simulation(
            sim_name=sim_name,
            parameters_file=parameters_file,
            cache=cache
        )

        print("\n Starting physics...(run_simulation)")

        solution = run_simulation(
            mesh=mesh,
            config=config,
            sim_name=sim_name,
            output_dir=base_output,
            parameter_filename=parameters_file,
        )

        print("\n Simulation completed")

    except Exception as e:

        import traceback
        import sys

        print("\n" + "=" * 70, flush=True)
        print(" SIMULATION FAILED", flush=True)
        print("=" * 70, flush=True)

        print(f"Simulation : {sim_name}", flush=True)
        print(f"Parameters : {parameters_file}", flush=True)
        print(f"Output dir : {base_output}", flush=True)

        print("\nException:", flush=True)
        print(f"  Type    : {type(e).__name__}", flush=True)
        print(f"  Message : {repr(e)}", flush=True)
        print(f"  Args    : {e.args}", flush=True)

        print("\nException traceback object:", flush=True)
        print(f"  {e.__traceback__}", flush=True)

        print("\nFull traceback:", flush=True)
        print("-" * 70, flush=True)

        traceback.print_exception(
            type(e),
            e,
            e.__traceback__,
            file=sys.stdout
        )

        print("-" * 70, flush=True)
        print("End traceback", flush=True)
        print("=" * 70, flush=True)

        raise
