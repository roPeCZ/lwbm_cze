from __future__ import annotations
import numpy.typing as npt
import numpy as np
from numba import njit


class LWBM:
    """
    Lumped Water Balance Rainfall-Runoff Model Class Wrapper.

    Args:
        prec (npt.NDArray[np.float64]): Array of daily mean precipitation
        temp (npt.NDArray[np.float64]): Array of daily mean temperature
        evap (npt.NDArray[np.float64]): Array of daily potential evapotranspiratino
        obs (npt.NDArray[np.float64]): Array of daily mean observed streamflow

    Returns:
        npt.NDArray[np.float64]: Modelled streamflow
    """

    __slots__ = "prec", "temp", "evap", "obs", "model"

    def __init__(
        self,
        prec: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
        evap: npt.NDArray[np.float64],
        obs: npt.NDArray[np.float64],
    ) -> None:

        self.prec = prec
        self.temp = temp
        self.evap = evap
        self.obs = obs
        self.model = model

    def __call__(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Call the lumped model njited function with provided parameters."""
        return self.model(params, self.prec, self.evap, self.temp)


@njit
def gamma_function(tp: float, m: float, duration: int) -> np.array:
    """
    An unit hydrograph function constructed via Gamma function.

    Args:
        tp (float): Lag-time of unit hydrograph culmination
        m (float): dimmensionless shape parameter
        duration (int): total number of days for which the hydrograph is evaluated

    Returns:
        npt.NDArray[np.float64]: Dimensionless coordinates of the specified unit hydrograph.
    """
    t = np.arange(duration)
    return np.exp(m) * (t / tp) ** m * (np.exp(-m * (t / tp)))


@njit
def gamma_coords(
    q_current: float,
    uh_q_coords: npt.NDArray[np.float64],
    uh_q_previous: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    It is used the mask the lumped surface and subsurface flow into the hydrograph.
    Args:
        q_current (float): Lumped value of evaluated streamflow for i-th day.
        uh_q_coords (npt.NDArray[np.float64]): Dimmensionless coordinates of the specified unit hydrograph
        uh_q_previous (npt.NDArray[np.float64]): Masked streamflow values over the unit hydrograph from previous time step

    Returns:
        npt.NDArray[np.float64]: Masked lumped streamflow value over the specified unit hydrograph.
                                 Current flow values from previous timestep are added.
    """
    return (q_current * uh_q_coords) + uh_q_previous


@njit
def model(
    parameters: npt.NDArray[np.float64],
    prec_values: npt.NDArray[np.float64],
    evap_values: npt.NDArray[np.float64],
    temp_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Lumped water balance rainfall-runoff model. Accounts for snowmelt using the 'degree-day' method
    and snow cover generation via defined interval for the division of precipitation into rainfall and snowfall.
    Surface and subsurface runoff is masked using two unit hydrographs which represents the quick flow part.
    Baseflow function follows the non-linear transform function from ARNO model, lags one timestep and represents the slow flow part.

    Args:
        parameters (npt.NDArray[np.float64]): Array of optimised model parameters
        prec_values (npt.NDArray[np.float64]): Array of daily mean precipitation
        evap_values (npt.NDArray[np.float64]): Array of daily mean potential evapotranspiration
        temp_values (npt.NDArray[np.float64]): Array of daily mean temperature

    Returns:
        npt.NDArray[np.float64]: Array of modelled streamflow
    """
    # Empty array initialisation
    q_overall = np.zeros(len(prec_values))

    # --- PARAMETERS SETUP ---+

    s_previous = parameters[0]  # Initial soil moisture [mm]
    sn_previous = parameters[1]  # Intial snow cover [mm]

    s_max = parameters[2]  # Maximum soil moisture [mm]
    s_min = parameters[3]  # Minimum soil moisture [mm]

    s_lim = (
        parameters[5] * s_max
    )  # [mm] With x_et [-] defines breakpoint of the ET function
    x_et = parameters[6]
    x_min = parameters[4]  # Slope soil function [-]

    t_crit = parameters[7]  # Snowmelt start [°C]
    t_lower = parameters[8]  # Lower bound of interval for snow+rain division [°C]
    t_upper = parameters[9]  # Upper bound of interval for snow+rain division [°C]

    qb_max = parameters[10]  # Maximum baseflow value [mm]
    alpha = parameters[11]  # Fraction of Qg_max [-]
    beta = parameters[12]  # Fraction of S_max [-]
    sb_lim = beta * s_max  # Defines breakpoint of the baseflow  function [mm]

    gamma_m_qs = parameters[13]  # Shape parameter of UH for surface flow
    gamma_m_qg = parameters[14]  # Shape parameter of UH for subsurface flow

    delay_qs = parameters[15]  # UH culmination lag-time for surface flow [-]

    # Subsurface flow lag time is assumed to be longer then lag-time of surface flow
    delay_qg = (
        parameters[16] * delay_qs
    )  # UH culmination lag-time for subsurface flow [-]

    # Initial unit hydrographs
    uh_coords_qs = gamma_function(delay_qs, gamma_m_qs, 10)
    uh_coords_qg = gamma_function(delay_qg, gamma_m_qg, 20)

    uh_qs_previous = np.zeros(len(uh_coords_qs))
    uh_qg_previous = np.zeros(len(uh_coords_qg))

    for i in range(len(prec_values)):
        ke = parameters[17]
        ks = parameters[18]
        ksn = parameters[19]
        kr = parameters[20] * ks
        expo = parameters[21]
        # --- DATA VALUES LOAD ---+

        t_current = temp_values[i]  # Daily temperature
        p_current = prec_values[i]  # Daily precipitation
        pet_current = evap_values[i]  # Daily potential evapotranspiration

        # --- SNOW ROUTINE ---+

        # Division coeff
        sn_coeff = (t_upper - t_current) / (t_upper - t_lower)
        if sn_coeff > 1.0:
            sn_coeff = 1.0
        elif sn_coeff < 0.0:
            sn_coeff = 0.0
        # Snow cover generation
        psn_current = p_current * sn_coeff
        pr_current = p_current - psn_current
        sn_current = sn_previous + psn_current
        # Surface and subsurface division adjustment during winter season with existing snow cover
        if sn_current:
            ks = kr
        # Snowmelt evaluation
        if t_current > t_crit:
            if sn_current:
                qsn_current = ksn * (t_current - t_crit)
                if qsn_current > sn_current:
                    qsn_current = sn_current
                    sn_current = 0.0
                else:
                    sn_current -= qsn_current
            else:
                qsn_current = 0.0
        else:
            qsn_current = 0.0

        # --- METEOROLOGICAL INPUT ---+

        # Total surface runoff
        pr_current += qsn_current
        # Hydrological losses estimation
        losses = ke * pet_current
        # Effective rainfall evaluation (generates runoff if positive)
        p_eff_current = pr_current - losses

        # --- EVAPOTRANSPIRATION ROUTINE ---+

        if pr_current >= pet_current:
            e_current = pet_current
        else:
            if s_previous <= s_lim:
                e_current = pr_current + (
                    (pet_current - pr_current) * x_et * s_previous / s_lim
                )
            else:
                e_current = pr_current + (
                    (pet_current - pr_current)
                    * (x_et + (1 - x_et) * (s_previous - s_lim) / (s_max - s_lim))
                )

        # --- SURFACE RUNOFF ROUTINE ---+

        qs_current = 0.0
        if p_eff_current > 0.0:
            if s_previous >= s_max:
                # If is soil layer saturated the whole effecive rainfall turns to runoff
                qs_current = p_eff_current
            else:
                qs_current = (
                    (s_previous * p_eff_current + 0.5 * p_eff_current**2)
                    * (1 - x_min)
                    / s_max
                )

        # --- SUBSURFACE RUNOFF ROUTINE ---+

        # Division to surface and subsurface runoff
        qg_current = qs_current * (1 - ks)
        qs_current *= ks
        # --- BASEFLOW ROUTINE ---+
        if (s_previous) <= sb_lim:
            baseflow_current = alpha * qb_max * s_previous / sb_lim
        else:
            baseflow_current = alpha * qb_max * s_previous / sb_lim + (
                qb_max - alpha * qb_max / beta
            ) * (((s_previous - sb_lim) / (s_max - sb_lim)) ** expo)

        # --- UNIT HYDROGRAPHS COORDINATES EVALUATION ---+

        uh_qs_current = gamma_coords(qs_current, uh_coords_qs, uh_qs_previous)
        uh_qg_current = gamma_coords(qg_current, uh_coords_qg, uh_qg_previous)

        # --- WATER BALANCE WITHIN SOIL LAYER ---+

        s_current = (
            s_previous
            + pr_current
            - e_current
            - qg_current
            - qs_current
            - baseflow_current
        )

        # Boundary check
        if s_current < s_min:
            s_current = s_min
        elif s_current > s_max:
            qs_current += s_current - s_max
            s_current = s_max
            uh_qs_current = gamma_coords(qs_current, uh_coords_qs, uh_qs_previous)

        # Save flow value into the initialised array
        q_overall[i] = uh_qs_current[0] + uh_qg_current[0] + baseflow_current

        # Redefine variables for the next timestep
        s_previous = s_current
        sn_previous = sn_current

        # Shift masked unit hydrographs since this timestep is over
        uh_qs_previous = np.roll(uh_qs_current, -1)
        uh_qg_previous = np.roll(uh_qg_current, -1)

        # roll function put first value to the end of array - needs to be replaced with 0
        uh_qs_previous[-1] = 0.0
        uh_qg_previous[-1] = 0.0

    return q_overall
