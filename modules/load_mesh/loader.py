from dolfinx.io import XDMFFile
from mpi4py import MPI

with XDMFFile(MPI.COMM_WORLD, xdmf_path, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

print("Topology dim:", mesh.topology.dim)
print("Geometry dim:", mesh.geometry.dim)
print("Num points:", mesh.geometry.x.shape)
