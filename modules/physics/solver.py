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


        self.xdmf.write_mesh(
            V.mesh
        )

        self.xdmf.write_function(
            self.u,
            self.t
        )

        source_field = self.get_source_field()

        if source_field is not None:
            self.xdmf.write_function(
                source_field,
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

    def get_source_field(self):
        """Return the active surface heat-flux field."""

        source = self.heat_eq.physics.get("source", {})

        if not source.get("enabled", False):
            return None

        source_type = source.get(
            "type",
            "gaussian"
        ).lower()

        if source_type == "gaussian":
            return self.heat_eq.q_gaussian

        elif source_type == "lamp":
            return self.heat_eq.q_lamp

        elif source_type == "laser":
            return self.heat_eq.q_laser

        else:
            raise ValueError(
                f"Unknown source type '{source_type}'"
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

        # --------------------------------------------------
        # Update source
        # --------------------------------------------------

        self.heat_eq.update_source(self.t)

        q = self.heat_eq.q_gaussian

        debug = False 
        if q is not None and debug:

            print(
                f"q_gaussian max = "
                f"{np.max(q.x.array):.12e}"
            )

            print(
                f"q_gaussian nonzero = "
                f"{np.count_nonzero(q.x.array)}"
            )

        # --------------------------------------------------
        # Assemble RHS
        # --------------------------------------------------

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

        #print(
        #    f"b norm = "
        #    f"{self.b.norm():.12e}"
        #)

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------

        u_before = self.u_n.x.array.copy()

        self.solver.solve(
            self.b,
            self.u.x.petsc_vec
        )

        reason = self.solver.getConvergedReason()

        #print(
        #    "PETSc reason:",
        #    reason
        #)

        #print(
        #    "Iterations :",
        #    self.solver.getIterationNumber()
        #)

        if reason < 0:

            print(
                self.solver.getResidualNorm()
            )

            raise RuntimeError(
                f"Solver failed ({reason})"
            )

        # --------------------------------------------------
        # Temperature change
        # --------------------------------------------------

        u_after = self.u.x.array

        du = u_after - u_before
        debug = False
        if debug:


            print(
                f"ΔT: min={du.min():.12e} K "
                f"max={du.max():.12e} K "
                f"mean={du.mean():.12e} K"
            )

        if np.isnan(u_after).any():

            raise ValueError(
                "NaN detected in solution"
            )

        # --------------------------------------------------
        # New simulation time
        # --------------------------------------------------

        new_time = (
            self.t
            + self.heat_eq.dt
        )

        # --------------------------------------------------
        # Energy balance
        #
        # IMPORTANT:
        # This must happen after the new temperature
        # has been solved, but before u_n is replaced.
        # --------------------------------------------------

        energy = self.heat_eq.energy_balance.update(
            self.u,
            new_time,
        )

        # --------------------------------------------------
        # Print energy balance
        # --------------------------------------------------

        if self.heat_eq.debug:

            print(
                "\n=== ENERGY BALANCE ==="
            )

            print(
                f"Source power       = "
                f"{energy['source_W']:.6e} W"
            )

            print(
                f"Convection loss    = "
                f"{energy['convection_W']:.6e} W"
            )

            print(
                f"Radiation loss     = "
                f"{energy['radiation_W']:.6e} W"
            )

            print(
                f"Net boundary power = "
                f"{energy['net_boundary_W']:.6e} W"
            )

            print(
                f"Source energy      = "
                f"{energy['source_J']:.6e} J"
            )

            print(
                f"Convection energy  = "
                f"{energy['convection_J']:.6e} J"
            )

            print(
                f"Radiation energy   = "
                f"{energy['radiation_J']:.6e} J"
            )

            print(
                f"Thermal energy     = "
                f"{energy['thermal_energy_J']:.6e} J"
            )

            print(
                f"Energy residual    = "
                f"{energy['energy_residual_J']:.6e} J"
            )

            print(
                "========================"
            )

        # --------------------------------------------------
        # Update previous solution
        # --------------------------------------------------

        self.u_n.x.array[:] = u_after

        # --------------------------------------------------
        # Advance simulation time
        # --------------------------------------------------

        self.t = new_time

        # --------------------------------------------------
        # Return energy information
        #
        # The caller can merge this dictionary into the
        # existing ROI/temperature CSV record.
        # --------------------------------------------------

        return energy

    # ======================================================
    # SAVE FRAME
    # ======================================================

    def save_frame(self):

        self.xdmf.write_function(
            self.u,
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
        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------

        self.point_tracker.sample(
            self.t,
            self.u,
        )

        # --------------------------------------------------
        # Time integration
        # --------------------------------------------------

        while self.t < T:

            # --------------------------------------------------
            # Solve one timestep
            # --------------------------------------------------

            energy = self.step()

            # --------------------------------------------------
            # Store ROI + energy information
            # --------------------------------------------------

            self.point_tracker.sample(
                self.t,
                self.u,
                energy=energy,
            )

            # --------------------------------------------------
            # Save visualization frame
            # --------------------------------------------------

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

        # --------------------------------------------------
        # Write ROI + energy CSV
        # --------------------------------------------------

        self.point_tracker.write_csv()
        self.xdmf.close()

        print("\n✅ Simulation finished")

        print(
            f"📄 Solution saved to:\n"
            f"{self.xdmf_path}"
        )

        return self.u
