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
from .gaussian import (
    initialize_gaussian,
    update_gaussian_source,
)

from .lamp import (
    initialize_lamp,
    update_lamp_source,
)

from .laser import (
    initialize_laser,
    update_laser_source,
)

from .energy_balance import EnergyBalance
class HeatEquation:
    """
    Transient heat equation with optional surface heat-flux sources.

        rho*c*dT/dt = div(k*grad(T))

    with optional boundary heat flux:

        - Gaussian surface heat flux [W/m²]
        - Lamp surface heat flux [W/m²]
        - Laser surface heat flux [W/m²]
        - convection
        - radiation (linearized)

    The incident heat flux is applied as a Neumann boundary
    condition on the illuminated surface.
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

        # Sources
        self.has_source = False
        self.has_convection = False
        self.has_radiation = False

        self.q_gaussian = None
        self.q_lamp = None
        self.q_laser = None

        self.ds_gaussian = None
        self.ds_lamp = None
        self.ds_laser = None

        source_cfg = self.physics.get(
            "source",
            {}
        )

        source_type = source_cfg.get(
            "type",
            "gaussian"
        ).lower()

        if source_type == "gaussian":

            initialize_gaussian(
                self,
                source_cfg
            )

        elif source_type == "lamp":

            initialize_lamp(
                self,
                source_cfg
            )

        elif source_type == "laser":

            initialize_laser(
                self,
                source_cfg
            )

        elif source_type:

            raise ValueError(
                f"Unknown source type '{source_type}'"
            )




        V_ufl = V.ufl_function_space()

        self.u = TrialFunction(V_ufl)
        self.v = TestFunction(V_ufl)
        self.energy_balance = EnergyBalance(self)
    # --------------------------------------------------
    # SOURCE UPDATE
    # --------------------------------------------------

    def update_source(self, t):

        source = self.physics.get(
            "source"
        )

        if (
            source is None
            or not source.get("enabled", False)
        ):

            if self.q_gaussian is not None:
                self.q_gaussian.x.array[:] = 0.0

            if self.q_lamp is not None:
                self.q_lamp.x.array[:] = 0.0

            if self.q_laser is not None:
                self.q_laser.x.array[:] = 0.0

            return

        source_type = source.get(
            "type",
            "gaussian"
        ).lower()

        # --------------------------------------------------
        # Gaussian surface source
        # --------------------------------------------------

        if source_type == "gaussian":

            update_gaussian_source(
                self,
                source,
                t,
            )

        # --------------------------------------------------
        # Lamp surface source
        # --------------------------------------------------

        elif source_type == "lamp":

            update_lamp_source(
                self,
                source,
                t,
            )

        # --------------------------------------------------
        # Laser surface source
        # --------------------------------------------------

        elif source_type == "laser":

            update_laser_source(
                self,
                source,
                t,
            )

        else:

            raise ValueError(
                f"Unknown source type '{source_type}'"
            )

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
        # Incident surface source
        # --------------------------------------------------

        source = self.physics.get(
            "source"
        )

        if source and source.get(
            "enabled",
            False
        ):

            self.has_source = True

            source_type = source.get(
                "type",
                "gaussian"
            ).lower()

            # --------------------------------------------------
            # Gaussian surface source
            # --------------------------------------------------

            if source_type == "gaussian":

                flux = self.q_gaussian

                L += (
                    dt
                    * flux
                    / (rho * c)
                    * v
                    * self.ds_gaussian(1)
                )

            # --------------------------------------------------
            # Lamp surface source
            # --------------------------------------------------

            elif source_type == "lamp":

                flux = self.q_lamp

                L += (
                    dt
                    * flux
                    / (rho * c)
                    * v
                    * self.ds_lamp(1)
                )

            # --------------------------------------------------
            # Laser surface source
            # --------------------------------------------------

            elif source_type == "laser":

                flux = self.q_laser

                L += (
                    dt
                    * flux
                    / (rho * c)
                    * v
                    * self.ds_laser(1)
                )

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

                print(
                    "   ✓ Gaussian incident surface flux"
                )

            elif source_type == "lamp":

                print(
                    "   ✓ Lamp incident surface flux"
                )

            elif source_type == "laser":

                print(
                    "   ✓ Laser incident Gaussian surface flux"
                )

        if self.has_convection:
            print("   ✓ Convection")

        if self.has_radiation:
            print("   ✓ Radiation")
