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
        """
        Update the active heat source according to the
        current simulation time.
        """

        source = self.physics.get("source")

        # ---------------------------------------------
        # Source disabled
        # ---------------------------------------------

        if source is None or not source.get("enabled", False):

            self.Q.x.array[:] = 0.0

            if hasattr(self, "q_flux"):
                self.q_flux.value = 0.0

            return

        # ---------------------------------------------
        # Dispatch by source type
        # ---------------------------------------------

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
