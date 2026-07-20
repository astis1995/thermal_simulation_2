import os
import subprocess
import tempfile
import meshio


def generate_geo(stl_path, geo_path, lc_min=0.01, lc_max=0.05):
    stl_path = os.path.abspath(stl_path)

    geo_content = f"""
Merge "{stl_path}";

// 🔥 Scale STL from mm → meters
Dilate {{0, 0, 0, 0.001}} {{
    Surface{{:}};
}};

ClassifySurfaces{{40*Pi/180, 1, 1}};
CreateGeometry;

Physical Surface(1) = Surface{{:}};

Mesh.CharacteristicLengthMin = {lc_min};
Mesh.CharacteristicLengthMax = {lc_max};

Mesh 2;
"""

    with open(geo_path, "w") as f:
        f.write(geo_content)


def run_gmsh(geo_path, msh_path):
    cmd = ["gmsh", geo_path, "-2", "-format", "msh2", "-o", msh_path]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("❌ Gmsh failed")


def convert_msh_to_xdmf_old(msh_path, output_prefix):
    mesh = meshio.read(msh_path)

    if "triangle" not in mesh.cells_dict:
        raise ValueError("❌ No triangle cells found")

    triangles = mesh.cells_dict["triangle"]

    xdmf_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", triangles)]
    )

    xdmf_path = output_prefix + ".xdmf"
    meshio.write(xdmf_path, xdmf_mesh)

    return xdmf_path


def convert_stl_to_xdmf(stl_path, output_path, lc_min=0.01, lc_max=0.05):
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"❌ STL not found: {stl_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        geo_path = os.path.join(tmpdir, "temp.geo")
        msh_path = os.path.join(tmpdir, "temp.msh")

        generate_geo(stl_path, geo_path, lc_min, lc_max)
        run_gmsh(geo_path, msh_path)

        xdmf_path = convert_msh_to_xdmf_old(msh_path, output_path)

    print(f"✅ Mesh generated: {xdmf_path}")
    return xdmf_path
