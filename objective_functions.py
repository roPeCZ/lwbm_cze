from __future__ import annotations
from typing import Callable
import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def nse_jit(
    modeled_data: npt.NDArray[np.float64], observed_data: npt.NDArray[np.float64]
) -> float:
    """
    Nash-Sutcliffe efficiency (NSE) model performance measure.
    Only the fraction part of the original formula is evaluated since it is desired it to be as small as possible.
    Both arrays from argument have to have same units of the streamflow (milimetres, since the LWBM uses these units)

    Args:
        modeled_data (npt.NDArray[np.float64]): Array of modelled streamflow
        observed_data (npt.NDArray[np.float64]): Array of observed streamflow

    Returns:
        Float: Value defining the error with modelled data. Subtracting this value from 1 will give true NSE value.
    """
    numerator = np.square(np.subtract(observed_data, modeled_data))
    denominator = np.square(np.subtract(observed_data, np.mean(observed_data)))
    return np.sum(numerator) / np.sum(denominator)


class ObjectiveFunctions:
    """
    Class for objective functions. It is initialised with model type and observed data,
    thus observed data variable is always available for the function.
    Any different performnace measures can be added/defined as a new method.
    """

    def __init__(self, model: Callable, observed_data: npt.NDArray[np.float64]) -> None:
        self.model = model
        self.observed_data = observed_data

    def nse(self, parameters: npt.NDArray[np.float64]) -> float:
        modeled_data = self.model(parameters)
        return nse_jit(modeled_data=modeled_data, observed_data=self.observed_data)
