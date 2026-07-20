# modules/physics/solver.py

import os
from datetime import datetime

import numpy as np

from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector
)
from .roi import PointTracker
from petsc4py import PETSc


class HeatSolver:



    def __init__(
        self,
        V,
        heat_equation,
        u_n,
        sim_name,
        output_dir,
        roi_config=None,
        t0=0.0,
    ):

        self.V = V
        self.heat_eq = heat_equation
        self.u_n = u_n
        self.t = float(t0)

        print("\n🧠 Initializing HeatSolver...")

        # ==================================================
        # SOLUTION FUNCTION
        # ==================================================

        self.u = fem.Function(V)
        self.u.name = "Temperature"
        self.u_c = fem.Function(V)
        self.u_c.name = "Temperature_C"
        # Copy initial condition into visualization field
        self.u.x.array[:] = self.u_n.x.array

        # ==================================================
        # OUTPUT DIRECTORY
        # ==================================================

        results_dir = os.path.join(
            output_dir,
            "results"
        )

        os.makedirs(
            results_dir,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        # ==================================================
        # PointTracker
        # ==================================================
        print("ROI CONFIG:")
        print(roi_config)

        self.point_tracker = PointTracker(
            mesh=V.mesh,
            function_space=V,
            config=roi_config,
            output_dir=results_dir,
            filename=f"{sim_name}-{timestamp}-roi.csv",
        )
        # ==================================================
        # XDMF OUTPUT
        # ==================================================

        self.xdmf_path = os.path.join(
            results_dir,
            f"{sim_name}-{timestamp}.xdmf"
        )

        self.xdmf = XDMFFile(
            V.mesh.comm,
            self.xdmf_path,
            "w"
        )


        # ==================================================
        # VARIATIONAL FORMS
        # ==================================================

        self.a, self.L = self.heat_eq.build_forms(
            self.u_n
        )

        self.a_form = fem.form(self.a)
        self.L_form = fem.form(self.L)

        print("   ✔ Forms compiled")

        # ==================================================
        # INITIAL OUTPUT
        # ==================================================

        self.heat_eq.update_source(self.t)

        self.xdmf.write_mesh(
            V.mesh
        )

        self.xdmf.write_function(
            self.u,
            self.t
        )

        if self.heat_eq.has_source:
            self.xdmf.write_function(
                self.heat_eq.Q,
                self.t
            )

        print(f"   📄 XDMF output: {self.xdmf_path}")

        # ==================================================
        # MATRIX
        # ==================================================

        self.A = assemble_matrix(
            self.a_form
        )

        self.A.assemble()
        print(self.A.getInfo())

        #zero_rows = self.A.findZeroRows()
        print("A norm =", self.A.norm())
        #print("Number of zero rows:", zero_rows.getSize())

        #if zero_rows.getSize() > 0:
        #    print("Zero rows:", zero_rows.getIndices())
        print("   ✔ Matrix assembled")
        print(
            f"   🔢 Matrix size: {self.A.getSize()}"
        )

        # ==================================================
        # RHS VECTOR
        # ==================================================

        self.b = self.A.createVecRight()

        print(
            "   ✔ RHS vector created"
        )

        # ==================================================
        # PETSc SOLVER
        # ==================================================

        self.solver = PETSc.KSP().create(
            V.mesh.comm
        )

        self.solver.setOperators(
            self.A
        )

        self.solver.setType(
            PETSc.KSP.Type.PREONLY
        )

        pc = self.solver.getPC()

        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

        pc = self.solver.getPC()

        print("KSP:", self.solver.getType())
        print("PC :", pc.getType())

        try:
            print("Factor solver:", pc.getFactorSolverType())
        except Exception as e:
            print("Factor solver: unknown", e)

        print(
            "   ✔ PETSc solver configured (LU)"
        )

        # ==================================================
        # INITIAL DIAGNOSTICS
        # ==================================================

        u0 = self.u_n.x.array

        print(
            f"   🔥 Initial Tmax: {u0.max():.6f}"
        )

        print(
            f"   🔥 Initial Tmin: {u0.min():.6f}"
        )

    # ======================================================
    # update celsius
    # ======================================================
    def update_celsius(self):
        """Update Celsius field from Kelvin solution."""
        self.u_c.x.array[:] = self.u.x.array - 273.15
    # ======================================================
    # STEP
    # ======================================================

    def step(self):

        self.heat_eq.update_source(self.t)

        with self.b.localForm() as loc:
            loc.set(0)

        assemble_vector(
            self.b,
            self.L_form
        )
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        print("b norm =", self.b.norm())
        self.solver.solve(
            self.b,
            self.u.x.petsc_vec
        )

        reason = self.solver.getConvergedReason()

        print("PETSc reason:", reason)
        print("Iterations :", self.solver.getIterationNumber())

        if reason < 0:

            print(self.solver.getResidualNorm())

            raise RuntimeError(
                f"Solver failed ({reason})"
            )

        u_arr = self.u.x.array

        if np.isnan(u_arr).any():

            raise ValueError(
                "NaN detected in solution"
            )

        # Update previous solution
        self.u_n.x.array[:] = u_arr

        self.t += self.heat_eq.dt
        self.point_tracker.sample(
            self.t,
            self.u,
        )
        return self.u

    # ======================================================
    # SAVE FRAME
    # ======================================================

    def save_frame(self):

        self.update_celsius()

        self.xdmf.write_function(
            self.u,
            self.t
        )

        self.xdmf.write_function(
            self.u_c,
            self.t
        )

        if self.heat_eq.has_source:
            self.xdmf.write_function(
                self.heat_eq.Q,
                self.t
            )

    # ======================================================
    # RUN
    # ======================================================

    def run(
        self,
        T,
        save_every=10
    ):

        step = 0

        num_steps = int(
            T / self.heat_eq.dt
        )

        print("\n🚀 Starting simulation:")
        print(
            f"   dt = {self.heat_eq.dt}"
        )
        print(
            f"   total steps = {num_steps}"
        )
        print(
            f"   save every = {save_every}"
        )
        #sample initial condition
        self.point_tracker.sample(
            self.t,
            self.u,
        )
        while self.t < T:

            self.step()

            if step % save_every == 0:

                self.save_frame()

                Tmax = float(
                    np.max(self.u.x.array)
                )

                Tmin = float(
                    np.min(self.u.x.array)
                )

                print(
                    f"t={self.t:.1f}s "
                    f"Tmin={Tmin:.3f} K "
                    f"Tmax={Tmax:.3f} K"
                )

            step += 1

        self.point_tracker.write_csv()
        self.xdmf.close()

        print("\n✅ Simulation finished")

        print(
            f"📄 Solution saved to:\n"
            f"{self.xdmf_path}"
        )

        return self.u
