# modules/physics/heat_equation.py

from dolfinx import fem
from ufl import TrialFunction, TestFunction, dx, dot, grad


class HeatEquation:
    """
    Pure PDE model:
        u_t - alpha Δu = Q

    Discretized with Backward Euler:
        (u - u_n) + dt * alpha Δu = dt * Q
    """

    def __init__(self, mesh, V, fields, dt, Q=None, debug=False):

        self.mesh = mesh
        self.V = V
        self.fields = fields
        self.dt = float(dt)
        self.debug = debug

        # Heat source
        if Q is None:
            self.Q = fem.Constant(mesh, 0.0)
            self.has_source = False
        else:
            self.Q = Q
            self.has_source = True

        # UFL spaces (CRITICAL: correct API)
        V_ufl = V.ufl_function_space()
        self.u = TrialFunction(V_ufl)
        self.v = TestFunction(V_ufl)

        if self.debug:
            self._print_info()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def build_forms(self, u_n):
        """
        Returns:
            a(u,v), L(v)
        """

        alpha = self.fields.alpha
        dt = self.dt
        Q = self.Q

        u = self.u
        v = self.v

        # ✅ Stable backward Euler form
        a = u * v * dx + dt * alpha * dot(grad(u), grad(v)) * dx
        L = u_n * v * dx + dt * Q * v * dx

        if self.debug:
            self._print_forms()

        return a, L

    # --------------------------------------------------
    # INTERNAL DEBUG
    # --------------------------------------------------
    def _print_info(self):
        print("\n🧠 HeatEquation initialized:")
        print(f"   dt = {self.dt}")
        print(f"   alpha = {float(self.fields.alpha.value):.6e}")
        print(f"   heat source = {'ON' if self.has_source else 'OFF'}")
        print(f"   mesh dims = topo:{self.mesh.topology.dim}, geo:{self.mesh.geometry.dim}")

    def _print_forms(self):
        print("\n📐 Building variational forms:")
        print("   a(u,v): mass + diffusion")
        print(f"   L(v): previous + {'source' if self.has_source else 'zero'}")
