from dolfinx import fem
from ufl import (
    TrialFunction,
    TestFunction,
    dx,
    ds,
    dot,
    grad,
)

from .fixed_heat import (
    initialize_fixed_heat,
    update_fixed_heat,
)

from .fixed_temperature import (
    initialize_fixed_temperature,
)

from .energy_balance import EnergyBalance


class HeatEquation:
    """
    Transient heat equation.

    rho*c*dT/dt = div(k*grad(T)) + Q

    Optional physics:

        - fixed_heat:
            volumetric heat generation [W/m^3]
            inside a rectangular prism.

        - fixed_temperature:
            Dirichlet temperature constraint [K]
            inside a rectangular prism.

        - convection:
            h [W/(m^2 K)]
            ambient temperature [K]

        - radiation:
            linearized radiation boundary condition.
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

        # --------------------------------------------------
        # PHYSICS CONFIGURATION
        # --------------------------------------------------

        self.physics = (
            config
            .get("simulation", {})
            .get("physics", {})
        )

        # --------------------------------------------------
        # FLAGS
        # --------------------------------------------------

        self.has_fixed_heat = False
        self.has_fixed_temperature = False
        self.has_convection = False
        self.has_radiation = False

        # --------------------------------------------------
        # FIXED HEAT
        # --------------------------------------------------

        self.q_fixed_heat = None
        self.dx_fixed_heat = None
        self.fixed_heat_volume = None
        self.fixed_heat_cells = None
        self.fixed_heat_tags = None

        fixed_heat_cfg = self.physics.get(
            "fixed_heat",
            {}
        )

        if fixed_heat_cfg.get("enabled", False):

            initialize_fixed_heat(
                self,
                fixed_heat_cfg,
            )

            self.has_fixed_heat = True

        # --------------------------------------------------
        # FIXED TEMPERATURE
        # --------------------------------------------------

        self.fixed_temperature_bcs = []
        self.fixed_temperature_dofs = None
        self.fixed_temperature_value = None
        self.fixed_temperature_constant = None

        fixed_temperature_cfg = self.physics.get(
            "fixed_temperature",
            {}
        )

        if fixed_temperature_cfg.get("enabled", False):

            initialize_fixed_temperature(
                self,
                fixed_temperature_cfg,
            )

            self.has_fixed_temperature = True

        # --------------------------------------------------
        # VARIATIONAL FUNCTIONS
        # --------------------------------------------------

        V_ufl = V.ufl_function_space()

        self.u = TrialFunction(V_ufl)
        self.v = TestFunction(V_ufl)

        # --------------------------------------------------
        # ENERGY BALANCE
        # --------------------------------------------------

        self.energy_balance = EnergyBalance(self)

        if self.debug:
            self._print_initialization()

    # ======================================================
    # UPDATE PHYSICS
    # ======================================================

    def update_physics(self, t):
        """
        Update all time-dependent physics.

        Currently:
            - fixed volumetric heat
        """

        if self.has_fixed_heat:

            update_fixed_heat(
                self,
                self.physics["fixed_heat"],
                t,
            )

    # ======================================================
    # BUILD VARIATIONAL FORMS
    # ======================================================

    def build_forms(self, u_n):

        alpha = self.fields.alpha

        rho = float(
            self.fields.rho.value
        )

        c = float(
            self.fields.c.value
        )

        dt = self.dt

        u = self.u
        v = self.v

        # --------------------------------------------------
        # BASE HEAT EQUATION
        # --------------------------------------------------

        a = (
            u * v * dx
            + dt
            * alpha
            * dot(grad(u), grad(v))
            * dx
        )

        L = (
            u_n * v * dx
        )

        # ==================================================
        # FIXED VOLUMETRIC HEAT
        # ==================================================

        if self.has_fixed_heat:

            if self.q_fixed_heat is None:
                raise RuntimeError(
                    "fixed_heat is enabled, but "
                    "q_fixed_heat was not initialized."
                )

            if self.dx_fixed_heat is None:
                raise RuntimeError(
                    "fixed_heat is enabled, but "
                    "dx_fixed_heat was not initialized."
                )

            L += (
                dt
                * self.q_fixed_heat
                / (rho * c)
                * v
                * self.dx_fixed_heat(1)
            )

        # ==================================================
        # CONVECTION
        # ==================================================

        convection = self.physics.get(
            "convection",
            {}
        )

        if convection.get("enabled", False):

            self.has_convection = True

            h = float(
                convection["h"]
            )

            T_inf = float(
                convection["ambient"]
            )

            if h < 0.0:
                raise ValueError(
                    "convection.h must be >= 0."
                )

            beta = h / (rho * c)

            a += (
                dt
                * beta
                * u
                * v
                * ds
            )

            L += (
                dt
                * beta
                * T_inf
                * v
                * ds
            )

        # ==================================================
        # RADIATION
        # ==================================================

        radiation = self.physics.get(
            "radiation",
            {}
        )

        if radiation.get("enabled", False):

            self.has_radiation = True

            emissivity = float(
                radiation["emissivity"]
            )

            T_inf = float(
                radiation["ambient"]
            )

            if not 0.0 <= emissivity <= 1.0:
                raise ValueError(
                    "radiation.emissivity must be between "
                    "0 and 1."
                )

            if T_inf < 0.0:
                raise ValueError(
                    "radiation.ambient must be >= 0 K."
                )

            # ----------------------------------------------
            # Linearized radiation coefficient
            #
            # q_rad ≈ h_rad (T - T_inf)
            #
            # h_rad = 4 eps sigma T_inf^3
            # ----------------------------------------------

            hr = (
                4.0
                * emissivity
                * self.SIGMA
                * T_inf**3
            )

            beta = hr / (rho * c)

            a += (
                dt
                * beta
                * u
                * v
                * ds
            )

            L += (
                dt
                * beta
                * T_inf
                * v
                * ds
            )

        # --------------------------------------------------
        # DEBUG
        # --------------------------------------------------

        if self.debug:
            self._print_info()

        return a, L

    # ======================================================
    # INITIALIZATION DIAGNOSTICS
    # ======================================================

    def _print_initialization(self):

        print("\n" + "=" * 60)
        print("HeatEquation initialization")
        print("=" * 60)

        print(
            f"dt                 = {self.dt:.6g} s"
        )

        print(
            f"alpha              = "
            f"{float(self.fields.alpha.value):.6e} m²/s"
        )

        print(
            f"fixed_heat         = "
            f"{'ON' if self.has_fixed_heat else 'OFF'}"
        )

        print(
            f"fixed_temperature = "
            f"{'ON' if self.has_fixed_temperature else 'OFF'}"
        )

        print(
            f"convection         = "
            f"{'ON' if self.physics.get('convection', {}).get('enabled', False) else 'OFF'}"
        )

        print(
            f"radiation          = "
            f"{'ON' if self.physics.get('radiation', {}).get('enabled', False) else 'OFF'}"
        )

        print(
            f"mesh               = "
            f"topo:{self.mesh.topology.dim}, "
            f"geo:{self.mesh.geometry.dim}"
        )

        print("=" * 60)

    # ======================================================
    # FORMULATION DIAGNOSTICS
    # ======================================================

    def _print_info(self):

        print("\nHeatEquation")

        print(
            f"   dt           = {self.dt}"
        )

        print(
            f"   alpha        = "
            f"{float(self.fields.alpha.value):.6e}"
        )

        print("\nVariational formulation")

        print("   ✓ Transient term")

        print("   ✓ Thermal diffusion")

        if self.has_fixed_heat:

            print(
                "   ✓ Fixed volumetric heat"
            )

            if self.fixed_heat_volume is not None:

                print(
                    f"     volume = "
                    f"{self.fixed_heat_volume:.6e} m³"
                )

        if self.has_fixed_temperature:

            print(
                "   ✓ Fixed temperature"
            )

            print(
                f"     T = "
                f"{self.fixed_temperature_value:.6f} K"
            )

            print(
                f"     DOFs = "
                f"{len(self.fixed_temperature_dofs)}"
            )

        if self.has_convection:

            convection = self.physics[
                "convection"
            ]

            print(
                "   ✓ Convection"
            )

            print(
                f"     h = "
                f"{float(convection['h']):.6g} W/(m² K)"
            )

            print(
                f"     T∞ = "
                f"{float(convection['ambient']):.6f} K"
            )

        if self.has_radiation:

            radiation = self.physics[
                "radiation"
            ]

            print(
                "   ✓ Radiation"
            )

            print(
                f"     emissivity = "
                f"{float(radiation['emissivity']):.6g}"
            )

            print(
                f"     T∞ = "
                f"{float(radiation['ambient']):.6f} K"
            )
