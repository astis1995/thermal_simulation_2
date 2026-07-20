from dolfinx import fem
from ufl import (
    TrialFunction,
    TestFunction,
    dx,
    ds,
    dot,
    grad,
)
import numpy as np
from .gaussian import update_gaussian_source
from .lamp import (
    initialize_lamp,
    update_lamp_source,
)

class HeatEquation:
    """
    Transient heat equation

        rho*c*dT/dt = div(k grad(T)) + Q

    with optional

        • volumetric heat source
        • convection
        • radiation (linearized)
    """

    SIGMA = 5.670374419e-8

    def __init__(
        self,
        mesh,
        V,
        fields,
        dt,
        config,
        debug=False,
    ):

        self.mesh = mesh
        self.V = V
        self.fields = fields
        self.dt = float(dt)
        self.debug = debug

        self.physics = (
            config
            .get("simulation", {})
            .get("physics", {})
        )

        # Spatial volumetric heat source (W/m³)
        self.Q = fem.Function(V)

        source_cfg = self.physics.get("source", {})

        if source_cfg.get("type") == "lamp":
            initialize_lamp(self, source_cfg)

        self.Q.name = source_cfg.get(
            "name",
            "HeatSource"
        )

        self.Q.x.array[:] = 0.0

        self.has_source = False
        self.has_convection = False
        self.has_radiation = False

        V_ufl = V.ufl_function_space()

        self.u = TrialFunction(V_ufl)
        self.v = TestFunction(V_ufl)

    # --------------------------------------------------
    # SOURCE UPDATE
    # --------------------------------------------------

    def update_source(self, t):

        source = self.physics.get("source")

        if source is None or not source.get("enabled", False):

            self.Q.x.array[:] = 0.0
            return

        source_type = source.get("type", "gaussian")

        if source_type == "gaussian":

            update_gaussian_source(
                self,
                source,
                t,
            )

        elif source_type == "lamp":

            update_lamp_source(
                self,
                source,
                t,
            )

        else:

            raise ValueError(
                f"Unknown source type '{source_type}'"
            )

    # --------------------------------------------------

    def build_forms(self, u_n):

        alpha = self.fields.alpha
        rho = float(self.fields.rho.value)
        c = float(self.fields.c.value)

        dt = self.dt

        u = self.u
        v = self.v

        # --------------------------------------------------
        # Base equation
        # --------------------------------------------------

        a = (
            u * v * dx
            + dt * alpha * dot(grad(u), grad(v)) * dx
        )

        L = (
            u_n * v * dx
        )

        # --------------------------------------------------
        # Heat source
        # --------------------------------------------------

        source = self.physics.get("source")

        if source and source.get("enabled", False):

            self.has_source = True

            source_type = source.get("type", "gaussian")

            if source_type == "gaussian":

                source_term = self.Q / (rho * c)

                L += dt * source_term * v * dx

            elif source_type == "lamp":

                flux = self.q_flux / (rho * c)

                L += dt * flux * v * self.ds_lamp(1)

            else:

                raise ValueError(
                    f"Unknown source type '{source_type}'"
                )
        # --------------------------------------------------
        # Convection
        # --------------------------------------------------

        convection = self.physics.get("convection")

        if convection and convection.get("enabled", False):

            self.has_convection = True

            h = float(convection["h"])
            T_inf = float(convection["ambient"])

            beta = h / (rho * c)

            a += dt * beta * u * v * ds
            L += dt * beta * T_inf * v * ds

        # --------------------------------------------------
        # Radiation (linearized)
        # --------------------------------------------------

        radiation = self.physics.get("radiation")

        if radiation and radiation.get("enabled", False):

            self.has_radiation = True

            emissivity = float(
                radiation["emissivity"]
            )

            T_inf = float(
                radiation["ambient"]
            )

            hr = (
                4.0
                * emissivity
                * self.SIGMA
                * T_inf**3
            )

            beta = hr / (rho * c)

            a += dt * beta * u * v * ds
            L += dt * beta * T_inf * v * ds

        if self.debug:
            self._print_info()

        return a, L

    # --------------------------------------------------

    def _print_info(self):

        print("\n🧠 HeatEquation")

        print(f"   dt           = {self.dt}")
        print(
            f"   alpha        = {float(self.fields.alpha.value):.6e}"
        )

        print(
            f"   source       = {'ON' if self.has_source else 'OFF'}"
        )

        print(
            f"   convection   = {'ON' if self.has_convection else 'OFF'}"
        )

        print(
            f"   radiation    = {'ON' if self.has_radiation else 'OFF'}"
        )

        print(
            f"   mesh         = topo:{self.mesh.topology.dim}, geo:{self.mesh.geometry.dim}"
        )

        print("\n📐 Variational formulation")

        print("   ✓ Diffusion")

        if self.has_source:

            source = self.physics.get("source", {})
            source_type = source.get("type", "gaussian")

            if source_type == "gaussian":
                print("   ✓ Gaussian volumetric source")

            elif source_type == "lamp":
                print("   ✓ Lamp boundary flux")

            else:
                print(f"   ✓ Source ({source_type})")

        if self.has_convection:
            print("   ✓ Convection")

        if self.has_radiation:
            print("   ✓ Radiation")
