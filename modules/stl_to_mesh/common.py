import os

import gmsh
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio


UNIT_SCALE = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
}


def initialize_gmsh(
    lc_min: float,
    lc_max: float,
):
    """Initialize Gmsh and configure global mesh options."""

    gmsh.initialize()

    gmsh.option.setNumber(
        "Geometry.Tolerance",
        1e-8,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        float(lc_min),
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        float(lc_max),
    )


def finalize_gmsh():
    """Finalize the Gmsh session."""

    gmsh.finalize()


def load_stl(stl_path: str):
    """Load an STL into the current Gmsh model."""

    if not os.path.exists(stl_path):
        raise FileNotFoundError(
            f"STL file not found: {stl_path}"
        )

    gmsh.merge(stl_path)

    bbox = gmsh.model.getBoundingBox(-1, -1)

    print("\nSTL bounding box (raw Gmsh)")
    print(f"X: {bbox[0]:.6e} -> {bbox[3]:.6e}")
    print(f"Y: {bbox[1]:.6e} -> {bbox[4]:.6e}")
    print(f"Z: {bbox[2]:.6e} -> {bbox[5]:.6e}")

    entities = gmsh.model.getEntities()

    if not entities:
        raise RuntimeError(
            "No entities imported from STL."
        )

    return entities


def convert_model_to_mesh(unit: str):
    """
    Convert the current Gmsh model to a DOLFINx mesh
    and scale coordinates to SI units.
    """

    scale = UNIT_SCALE.get(unit.lower())

    if scale is None:
        raise ValueError(
            f"Unsupported unit '{unit}'."
        )

    mesh_data = gmshio.model_to_mesh(
        gmsh.model,
        MPI.COMM_WORLD,
        0,
    )

    mesh = mesh_data.mesh

    if scale != 1.0:
        mesh.geometry.x[:] *= scale


    xyz_min = mesh.geometry.x.min(axis=0)
    xyz_max = mesh.geometry.x.max(axis=0)
    size = xyz_max - xyz_min

    print("\n==============================")
    print("Geometry after unit conversion")
    print("==============================")
    print(f"Unit           : m")
    print(f"Scale applied  : {scale}")
    print(f"X: {xyz_min[0]:.6e} -> {xyz_max[0]:.6e}")
    print(f"Y: {xyz_min[1]:.6e} -> {xyz_max[1]:.6e}")
    print(f"Z: {xyz_min[2]:.6e} -> {xyz_max[2]:.6e}")
    print()
    print(f"Lx = {size[0]:.6e} m")
    print(f"Ly = {size[1]:.6e} m")
    print(f"Lz = {size[2]:.6e} m")

    print("\n==============================")
    print("DOLFINx mesh topology")
    print("==============================")
    print(f"Topological dimension: {mesh.topology.dim}")
    print(f"Geometrical dimension: {mesh.geometry.dim}")

    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        n = mesh.topology.index_map(dim).size_local
        print(f"Dimension {dim}: {n} entities")
    return mesh



def write_xdmf(
    mesh,
    filename: str,
):
    """Write a DOLFINx mesh to XDMF."""

    with XDMFFile(
        MPI.COMM_WORLD,
        filename,
        "w",
    ) as xdmf:

        xdmf.write_mesh(mesh)


def print_mesh_info(mesh):
    """Print basic mesh diagnostics."""

    xyz_min = mesh.geometry.x.min(axis=0)
    xyz_max = mesh.geometry.x.max(axis=0)

    print("\nMesh summary")
    print("---------------------------")
    print(f"Nodes: {mesh.geometry.x.shape[0]}")
    print(f"Topology dimension: {mesh.topology.dim}")
    print(f"Geometry dimension: {mesh.geometry.dim}")
    print(
        f"X: {xyz_min[0]:.6e} -> {xyz_max[0]:.6e}"
    )
    print(
        f"Y: {xyz_min[1]:.6e} -> {xyz_max[1]:.6e}"
    )
    print(
        f"Z: {xyz_min[2]:.6e} -> {xyz_max[2]:.6e}"
    )
