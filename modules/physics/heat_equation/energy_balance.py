"""
Energy balance and thermal energy accounting.

This module is intentionally independent from HeatEquation.

It tracks:

    Source power                  [W]
    Convective heat loss          [W]
    Radiative heat loss           [W]

    Source energy                 [J]
    Convective energy loss        [J]
    Radiative energy loss         [J]

    Stored thermal energy         [J]
    Change in stored energy       [J]

    Energy balance residual       [J]

Sign convention:

    source_W       > 0 : energy entering the body
    convection_W   > 0 : energy leaving by convection
    radiation_W    > 0 : energy leaving by radiation

Negative convection/radiation values mean that the
environment is supplying energy to the body.

For each timestep:

    E_source_step     = source_W * dt
    E_convection_step = convection_W * dt
    E_radiation_step  = radiation_W * dt

Energy balance:

    E_source
    - E_convection
    - E_radiation
    = Delta_U + residual

The source power is evaluated using the source state
already established by HeatEquation.update_source(t).

Convection and radiation are evaluated using the newly
solved temperature T(n+1), consistent with the implicit
Backward-Euler formulation.
"""

import csv

from dolfinx import fem
from ufl import ds, dx


class EnergyBalance:
    """
    Energy accounting for a HeatEquation instance.

    The class does not modify the thermal equation.

    It only reads:

        - material properties
        - source fields
        - boundary measures
        - temperature field
        - simulation timestep
    """

    SIGMA = 5.670374419e-8

    def __init__(self, heat_eq):

        self.heat_eq = heat_eq

        # --------------------------------------------------
        # Cumulative energies
        # --------------------------------------------------

        self.source_energy_J = 0.0

        self.convection_energy_J = 0.0

        self.radiation_energy_J = 0.0

        # --------------------------------------------------
        # Initial thermal energy
        #
        # This MUST be initialized before the first
        # timestep is solved.
        # --------------------------------------------------

        self.initial_thermal_energy_J = None

        # --------------------------------------------------
        # Previous thermal energy
        #
        # Used to calculate Delta U for each timestep.
        # --------------------------------------------------

        self.previous_thermal_energy_J = None

        self.previous_time_s = None

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        self.history = []

    # ======================================================
    # MPI HELPER
    # ======================================================

    def _global_sum(self, value):
        """
        Sum a locally assembled quantity across MPI ranks.
        """

        comm = self.heat_eq.mesh.comm

        return comm.allreduce(
            float(value)
        )

    # ======================================================
    # INITIALIZE
    # ======================================================

    def initialize(self, T):
        """
        Initialize the energy reference state.

        This must be called BEFORE the first timestep
        is solved.

        Parameters
        ----------
        T
            Initial temperature Function.
        """

        initial_energy = self.thermal_energy(T)

        self.initial_thermal_energy_J = (
            initial_energy
        )

        self.previous_thermal_energy_J = (
            initial_energy
        )

        self.previous_time_s = None

        self.source_energy_J = 0.0
        self.convection_energy_J = 0.0
        self.radiation_energy_J = 0.0

        self.history = []

    # ======================================================
    # SOURCE POWER
    # ======================================================

    def source_power(self):
        """
        Calculate total source power entering the body.

        Returns
        -------
        float
            Source power [W].
        """

        h = self.heat_eq

        source = h.physics.get("source")

        if not source:
            return 0.0

        if not source.get("enabled", False):
            return 0.0

        source_type = source.get(
            "type",
            "gaussian",
        ).lower()

        # --------------------------------------------------
        # Select source field and boundary measure
        # --------------------------------------------------

        if source_type == "gaussian":

            flux = h.q_gaussian
            measure = h.ds_gaussian

        elif source_type == "lamp":

            flux = h.q_lamp
            measure = h.ds_lamp

        elif source_type == "laser":

            flux = h.q_laser
            measure = h.ds_laser

        else:

            raise ValueError(
                f"Unknown source type '{source_type}'"
            )

        if flux is None:
            return 0.0

        if measure is None:
            return 0.0

        # --------------------------------------------------
        # Surface integral
        #
        # W/m² * m² = W
        # --------------------------------------------------

        local_power = fem.assemble_scalar(
            fem.form(
                flux * measure(1)
            )
        )

        return self._global_sum(
            local_power
        )

    # ======================================================
    # CONVECTION POWER
    # ======================================================

    def convection_power(self, T):
        """
        Calculate total convective heat loss.

        Positive:
            body loses heat to surroundings.

        Negative:
            surroundings heat the body.

        Returns
        -------
        float
            Convective power [W].
        """

        h = self.heat_eq

        convection = h.physics.get(
            "convection"
        )

        if not convection:
            return 0.0

        if not convection.get(
            "enabled",
            False,
        ):
            return 0.0

        h_conv = float(
            convection["h"]
        )

        T_inf = float(
            convection["ambient"]
        )

        # --------------------------------------------------
        # Convective heat flux
        #
        # q_conv = h (T - T_inf)
        #
        # W/m²
        # --------------------------------------------------

        q_conv = (
            h_conv
            * (T - T_inf)
        )

        # --------------------------------------------------
        # Surface integral
        #
        # W/m² * m² = W
        # --------------------------------------------------

        local_power = fem.assemble_scalar(
            fem.form(
                q_conv * ds
            )
        )

        return self._global_sum(
            local_power
        )

    # ======================================================
    # RADIATION POWER
    # ======================================================

    def radiation_power(self, T):
        """
        Calculate total radiative heat loss.

        Uses the same linearized radiation model
        implemented in HeatEquation:

            h_r = 4 epsilon sigma T_inf^3

            q_rad = h_r (T - T_inf)

        Positive:
            body loses heat by radiation.

        Negative:
            surroundings heat the body.

        Returns
        -------
        float
            Radiative power [W].
        """

        h = self.heat_eq

        radiation = h.physics.get(
            "radiation"
        )

        if not radiation:
            return 0.0

        if not radiation.get(
            "enabled",
            False,
        ):
            return 0.0

        emissivity = float(
            radiation["emissivity"]
        )

        T_inf = float(
            radiation["ambient"]
        )

        # --------------------------------------------------
        # Linearized radiation coefficient
        # --------------------------------------------------

        hr = (
            4.0
            * emissivity
            * self.SIGMA
            * T_inf**3
        )

        # --------------------------------------------------
        # Radiative heat flux
        #
        # W/m²
        # --------------------------------------------------

        q_rad = (
            hr
            * (T - T_inf)
        )

        # --------------------------------------------------
        # Surface integral
        #
        # W/m² * m² = W
        # --------------------------------------------------

        local_power = fem.assemble_scalar(
            fem.form(
                q_rad * ds
            )
        )

        return self._global_sum(
            local_power
        )

    # ======================================================
    # THERMAL ENERGY
    # ======================================================

    def thermal_energy(self, T):
        """
        Calculate total thermal energy represented
        by the temperature field.

            U = integral(rho * c * T dV)

        Returns
        -------
        float
            Thermal energy [J].
        """

        h = self.heat_eq

        rho = float(
            h.fields.rho.value
        )

        c = float(
            h.fields.c.value
        )

        local_energy = fem.assemble_scalar(
            fem.form(
                rho
                * c
                * T
                * dx
            )
        )

        return self._global_sum(
            local_energy
        )

    # ======================================================
    # UPDATE
    # ======================================================

    def update(self, T, t):
        """
        Calculate and record the energy balance
        for one completed timestep.

        Parameters
        ----------
        T
            Newly solved temperature Function T(n+1).

        t
            New simulation time t(n+1) [s].

        Returns
        -------
        dict
            Energy balance record.
        """

        t = float(t)

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if self.initial_thermal_energy_J is None:

            raise RuntimeError(
                "EnergyBalance has not been initialized. "
                "Call energy_balance.initialize(T_initial) "
                "before the first timestep."
            )

        # --------------------------------------------------
        # Current powers
        #
        # Source:
        #     source state was established by
        #     HeatEquation.update_source()
        #
        # Convection/radiation:
        #     evaluated using T(n+1)
        # --------------------------------------------------

        source_W = self.source_power()

        convection_W = (
            self.convection_power(T)
        )

        radiation_W = (
            self.radiation_power(T)
        )

        # --------------------------------------------------
        # Current stored thermal energy
        # --------------------------------------------------

        thermal_energy_J = (
            self.thermal_energy(T)
        )

        # --------------------------------------------------
        # Determine timestep
        # --------------------------------------------------

        if self.previous_time_s is None:

            # First completed timestep.
            #
            # The HeatSolver calls update() with
            # t = t_initial + dt.
            #
            # Therefore use the HeatEquation timestep.
            dt = self.heat_eq.dt

        else:

            dt = (
                t
                - self.previous_time_s
            )

        if dt < 0.0:

            raise ValueError(
                "EnergyBalance time moved backwards: "
                f"{self.previous_time_s} -> {t}"
            )

        # --------------------------------------------------
        # Energy entering/leaving during this timestep
        #
        # IMPORTANT:
        #
        # Do NOT use trapezoidal integration here.
        #
        # The PDE is Backward Euler:
        #
        #     T(n) -> T(n+1)
        #
        # and the source state used by the PDE was
        # established at the beginning of this timestep.
        #
        # Therefore:
        #
        #     E = P * dt
        # --------------------------------------------------

        source_energy_step_J = (
            source_W * dt
        )

        convection_energy_step_J = (
            convection_W * dt
        )

        radiation_energy_step_J = (
            radiation_W * dt
        )

        # --------------------------------------------------
        # Accumulate energy
        # --------------------------------------------------

        self.source_energy_J += (
            source_energy_step_J
        )

        self.convection_energy_J += (
            convection_energy_step_J
        )

        self.radiation_energy_J += (
            radiation_energy_step_J
        )

        # --------------------------------------------------
        # Change in stored thermal energy
        # --------------------------------------------------

        delta_U_step_J = (
            thermal_energy_J
            - self.previous_thermal_energy_J
        )

        delta_U_J = (
            thermal_energy_J
            - self.initial_thermal_energy_J
        )

        # --------------------------------------------------
        # Cumulative energy residual
        #
        # E_in - E_out - Delta_U
        # --------------------------------------------------

        residual_J = (
            self.source_energy_J
            - self.convection_energy_J
            - self.radiation_energy_J
            - delta_U_J
        )

        # --------------------------------------------------
        # Net boundary power
        # --------------------------------------------------

        net_boundary_W = (
            source_W
            - convection_W
            - radiation_W
        )

        # --------------------------------------------------
        # Net energy during this timestep
        # --------------------------------------------------

        net_boundary_energy_step_J = (
            source_energy_step_J
            - convection_energy_step_J
            - radiation_energy_step_J
        )

        # --------------------------------------------------
        # Per-step residual
        #
        # This is particularly useful for debugging.
        # --------------------------------------------------

        step_residual_J = (
            net_boundary_energy_step_J
            - delta_U_step_J
        )

        # --------------------------------------------------
        # Record
        # --------------------------------------------------

        record = {

            "time_s": t,

            # ----------------------------------------------
            # Instantaneous power
            # ----------------------------------------------

            "source_W": source_W,

            "convection_W": convection_W,

            "radiation_W": radiation_W,

            "net_boundary_W": net_boundary_W,

            # ----------------------------------------------
            # Energy added/lost during this timestep
            # ----------------------------------------------

            "source_step_J": (
                source_energy_step_J
            ),

            "convection_step_J": (
                convection_energy_step_J
            ),

            "radiation_step_J": (
                radiation_energy_step_J
            ),

            "net_boundary_step_J": (
                net_boundary_energy_step_J
            ),

            # ----------------------------------------------
            # Cumulative energy
            # ----------------------------------------------

            "source_J": (
                self.source_energy_J
            ),

            "convection_J": (
                self.convection_energy_J
            ),

            "radiation_J": (
                self.radiation_energy_J
            ),

            # ----------------------------------------------
            # Thermal energy
            # ----------------------------------------------

            "thermal_energy_J": (
                thermal_energy_J
            ),

            "delta_U_step_J": (
                delta_U_step_J
            ),

            "delta_U_J": (
                delta_U_J
            ),

            # ----------------------------------------------
            # Energy conservation
            # ----------------------------------------------

            "energy_residual_step_J": (
                step_residual_J
            ),

            "energy_residual_J": (
                residual_J
            ),
        }

        # --------------------------------------------------
        # Store history
        # --------------------------------------------------

        self.history.append(
            record
        )

        # --------------------------------------------------
        # Save state for next timestep
        # --------------------------------------------------

        self.previous_time_s = t

        self.previous_thermal_energy_J = (
            thermal_energy_J
        )

        return record

    # ======================================================
    # LATEST
    # ======================================================

    def latest(self):
        """
        Return the latest energy balance record.
        """

        if not self.history:
            return None

        return self.history[-1]

    # ======================================================
    # PRINT
    # ======================================================

    def print_latest(self):
        """
        Print the latest energy balance.
        """

        record = self.latest()

        if record is None:

            print(
                "No energy balance data recorded."
            )

            return

        #print(
        #    "\n=== ENERGY BALANCE ==="
        #)

        #print(
        #    f"Time                  = "
        #    f"{record['time_s']:.6f} s"
        #)

        #print(
        #    "\nInstantaneous power:"
        #)

        #print(
        #    f"  Source              = "
        #    f"{record['source_W']:.12e} W"
        #)

        #print(
        #    f"  Convection          = "
        #    f"{record['convection_W']:.12e} W"
        #)

        #print(
        #    f"  Radiation           = "
        #    f"{record['radiation_W']:.12e} W"
        #)

        #print(
        #    f"  Net boundary        = "
        #    f"{record['net_boundary_W']:.12e} W"
        #)

        #print(
        #    "\nThis timestep:"
        #)

        #print(
        #    f"  Source              = "
        #    f"{record['source_step_J']:.12e} J"
        #)

        #print(
        #    f"  Convection          = "
        #    f"{record['convection_step_J']:.12e} J"
        #)

        #print(
        #    f"  Radiation           = "
        #    f"{record['radiation_step_J']:.12e} J"
        #)

        #print(
        #    f"  Net boundary        = "
        #    f"{record['net_boundary_step_J']:.12e} J"
        #)

        #print(
        #    f"  Delta U             = "
        #    f"{record['delta_U_step_J']:.12e} J"
        #)

        #print(
        #    f"  Step residual       = "
        #    f"{record['energy_residual_step_J']:.12e} J"
        #)

        #print(
        #    "\nCumulative energy:"
        #)

        #print(
        #    f"  Source              = "
        #    f"{record['source_J']:.12e} J"
        #)

        #print(
        #    f"  Convection          = "
        #    f"{record['convection_J']:.12e} J"
        #)

        #print(
        #    f"  Radiation           = "
        #    f"{record['radiation_J']:.12e} J"
        #)

        #print(
        #    f"  Stored thermal      = "
        #    f"{record['thermal_energy_J']:.12e} J"
        #)

        #print(
        #    f"  Delta U             = "
        #    f"{record['delta_U_J']:.12e} J"
        #)

        #print(
        #    "\nEnergy conservation:"
        #)

        #print(
        #    f"  Residual            = "
        #    f"{record['energy_residual_J']:.12e} J"
        #)

        #print(
        #    "========================"
        #)

    # ======================================================
    # CSV EXPORT
    # ======================================================

    def save_csv(self, filename):
        """
        Save complete energy history to CSV.
        """

        if not self.history:

            raise RuntimeError(
                "No energy balance data to save."
            )

        fieldnames = list(
            self.history[0].keys()
        )

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                self.history
            )
