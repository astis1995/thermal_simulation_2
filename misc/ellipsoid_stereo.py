"""
ellipsoid_stereo.py

Generate a hollow upper ellipsoid using Gmsh + OpenCASCADE.

Workflow

    Unit sphere
          ↓
    Dilate -> outer ellipsoid
          ↓
    Unit sphere
          ↓
    Dilate -> inner ellipsoid
          ↓
    Boolean difference
          ↓
    Cut with z = 0
          ↓
    Mesh
          ↓
    Export STEP/STL/MSH
"""
import argparse
import gmsh

from pathlib import Path


class EllipsoidStereo:

    def __init__(
        self,
        filename="ellipsoid",
        major_axis=0.10,
        minor_axis=0.06,
        height=0.02,
        thickness=0.002,
        mesh_size=0.001,
        show_gui=True,
    ):

        self.filename = Path(filename).with_suffix("")

        self.A = major_axis / 2.0
        self.B = minor_axis / 2.0
        self.C = height

        self.thickness = thickness
        self.mesh_size = mesh_size
        self.show_gui = show_gui

        if thickness <= 0:
            raise ValueError("Thickness must be positive.")

        if thickness >= min(self.A, self.B, self.C):
            raise ValueError(
                "Thickness is larger than one of the semi-axes."
            )

    # -------------------------------------------------------

    def initialize(self):

        gmsh.initialize()

        gmsh.model.add("ellipsoid")

        gmsh.option.setNumber(
            "General.Terminal",
            1
        )

    # -------------------------------------------------------

    def finalize(self):

        gmsh.finalize()

    # -------------------------------------------------------

    def create_outer_ellipsoid(self):
        """
        Creates a unit sphere.

        Later we will dilate it into an ellipsoid.
        """

        sphere = gmsh.model.occ.addSphere(
            0.0,
            0.0,
            0.0,
            1.0,
        )

        return sphere

    # -------------------------------------------------------

    def create_inner_ellipsoid(self):
        """
        Creates the inner unit sphere.
        """

        sphere = gmsh.model.occ.addSphere(
            0.0,
            0.0,
            0.0,
            1.0,
        )

        return sphere

    # -------------------------------------------------------

    def scale_outer(self, tag):
        """
        Transform the unit sphere into the outer ellipsoid.
        """

        gmsh.model.occ.dilate(
            [(3, tag)],
            0.0, 0.0, 0.0,
            self.A,
            self.B,
            self.C,
        )

        return tag


    # -------------------------------------------------------

    def scale_inner(self, tag):
        """
        Transform the unit sphere into the inner ellipsoid.
        """

        Ai = self.A - self.thickness
        Bi = self.B - self.thickness
        Ci = self.C - self.thickness

        gmsh.model.occ.dilate(
            [(3, tag)],
            0.0, 0.0, 0.0,
            Ai,
            Bi,
            Ci,
        )

        return tag

    # -------------------------------------------------------

    def build_shell(self, outer, inner):
        """
        shell = outer - inner
        """

        shell, _ = gmsh.model.occ.cut(
            [(3, outer)],
            [(3, inner)],
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()

        if len(shell) != 1:
            raise RuntimeError(
                f"Expected one shell volume, got {shell}"
            )

        return shell[0][1]
    # -------------------------------------------------------

    def cut_upper_half(self, shell):
        """
        Intersect the shell with a box above z = 0.
        """

        box = gmsh.model.occ.addBox(
            -self.A,
            -self.B,
            0.0,
            2.0 * self.A,
            2.0 * self.B,
            self.C,
        )

        upper, _ = gmsh.model.occ.intersect(
            [(3, shell)],
            [(3, box)],
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()

        if len(upper) != 1:
            raise RuntimeError(
                f"Intersection failed: {upper}"
            )

        return upper[0][1]

    # -------------------------------------------------------

    def mesh(self):

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            self.mesh_size,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            self.mesh_size,
        )

        gmsh.option.setNumber(
            "Mesh.Algorithm3D",
            10,
        )  # HXT

        gmsh.option.setNumber(
            "Mesh.Optimize",
            1,
        )

        gmsh.option.setNumber(
            "Mesh.OptimizeNetgen",
            1,
        )

        gmsh.option.setNumber(
            "Mesh.SaveAll",
            1,
        )

        gmsh.model.mesh.generate(3)

    # -------------------------------------------------------

    def export(self):
        """
        Export CAD and STL.

        The STL is generated from a fresh surface mesh (2D),
        not from the tetrahedral mesh.
        """

        base = str(self.filename)

        #
        # STEP / BREP
        #

        gmsh.write(base + ".step")
        gmsh.write(base + ".brep")

        #
        # Remove any existing mesh
        #

        gmsh.model.mesh.clear()

        #
        # Generate only a surface mesh
        #

        gmsh.model.mesh.generate(2)

        gmsh.write(base + ".stl")

        #
        # Optional: regenerate the volume mesh
        #

        gmsh.model.mesh.clear()

        gmsh.model.mesh.generate(3)

        gmsh.write(base + ".msh")

        print(f"Exported {base}.step")
        print(f"Exported {base}.brep")
        print(f"Exported {base}.stl")
        print(f"Exported {base}.msh")

    # -------------------------------------------------------

    def generate(self):

        self.initialize()

        #
        # Geometry
        #

        outer = self.create_outer_ellipsoid()
        gmsh.model.occ.synchronize()

        inner = self.create_inner_ellipsoid()
        gmsh.model.occ.synchronize()

        outer = self.scale_outer(outer)
        gmsh.model.occ.synchronize()

        inner = self.scale_inner(inner)
        gmsh.model.occ.synchronize()

        shell = self.build_shell(
            outer,
            inner,
        )

        gmsh.model.occ.synchronize()

        shell = self.cut_upper_half(shell)

        gmsh.model.occ.synchronize()

        #
        # Physical groups
        #

        self.create_physical_groups(shell)

        #
        # Mesh
        #

        self.mesh()

        #
        # Export
        #

        self.export()

        if self.show_gui and gmsh.fltk.isAvailable():
            gmsh.fltk.run()

        self.finalize()

    def create_physical_groups(self, volume):

        #
        # Volume
        #

        gmsh.model.addPhysicalGroup(
            3,
            [volume],
            1,
        )

        gmsh.model.setPhysicalName(
            3,
            1,
            "Shell",
        )

        #
        # Surfaces
        #

        surfaces = gmsh.model.getBoundary(
            [(3, volume)],
            oriented=False,
            recursive=False,
        )

        outer = None
        inner = None
        bottom = None

        max_area = -1.0
        min_area = 1e100

        for dim, tag in surfaces:

            xmin, ymin, zmin, xmax, ymax, zmax = \
                gmsh.model.getBoundingBox(dim, tag)

            #
            # Bottom plane
            #

            if abs(zmax) < 1e-8 and abs(zmin) < 1e-8:

                bottom = tag
                continue

            #
            # Approximate area from bounding box
            #

            area = (
                (xmax - xmin)
                *
                (ymax - ymin)
            )

            if area > max_area:

                max_area = area
                outer = tag

            if area < min_area:

                min_area = area
                inner = tag

        if outer is not None:

            gmsh.model.addPhysicalGroup(
                2,
                [outer],
                11,
            )

            gmsh.model.setPhysicalName(
                2,
                11,
                "Outer",
            )

        if inner is not None:

            gmsh.model.addPhysicalGroup(
                2,
                [inner],
                12,
            )

            gmsh.model.setPhysicalName(
                2,
                12,
                "Inner",
            )

        if bottom is not None:

            gmsh.model.addPhysicalGroup(
                2,
                [bottom],
                13,
            )

            gmsh.model.setPhysicalName(
                2,
                13,
                "Bottom",
            )

        print()

        print("Physical groups created.")
        print("Outer :", outer)
        print("Inner :", inner)
        print("Bottom:", bottom)
# ===========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a hollow upper ellipsoid using Gmsh/OpenCASCADE."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="ellipsoid",
        help="Output filename (without extension).",
    )

    parser.add_argument(
        "--major",
        type=float,
        default=0.10,
        help="Major axis (m).",
    )

    parser.add_argument(
        "--minor",
        type=float,
        default=0.06,
        help="Minor axis (m).",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=0.02,
        help="Ellipsoid height (m).",
    )

    parser.add_argument(
        "--thickness",
        type=float,
        default=0.002,
        help="Shell thickness (m).",
    )

    parser.add_argument(
        "--mesh-size",
        type=float,
        default=0.001,
        help="Target mesh size (m).",
    )

    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Do not open the Gmsh GUI.",
    )

    args = parser.parse_args()

    model = EllipsoidStereo(
        filename=args.output,
        major_axis=args.major,
        minor_axis=args.minor,
        height=args.height,
        thickness=args.thickness,
        mesh_size=args.mesh_size,
        show_gui=not args.nogui,
    )

    if args.nogui:
        gmsh.fltk.initialize = lambda: None

    model.generate()
