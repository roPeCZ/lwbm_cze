# --- IMPORT FROM PYTHON LIBRARIES ---+

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

# --- IMPORT FROM LOCAL FILES ---+

from de import differentital_evolution
from objective_functions import ObjectiveFunctions
from lwbm import LWBM


@dataclass
class Parameter:
    """Representing each model parameter."""

    name: str
    lower_bound: float
    upper_bound: float
    description: str


def load_params_bounds_json() -> list[tuple[float, float]]:
    """
    Loads model parameter bounds into list of tuples from JSON file.

    Returns:
        list[tuple[float, float]]: List of lower and upper bounds from model parameters
    """
    with open("parameter_bounds.json", "r") as data_file:
        params_data: dict[str, Parameter] = json.load(data_file)
        model_parameters = {
            param: Parameter(
                name=param,
                lower_bound=params_data[param]["lower_bound"],
                upper_bound=params_data[param]["upper_bound"],
                description=params_data[param]["description"],
            )
            for param in params_data.keys()
        }

    return [
        (parameter.lower_bound, parameter.upper_bound)
        for parameter in model_parameters.values()
    ]


if __name__ == "__main__":

    # --- DATA LOAD ---+

    calibration_data = pd.read_excel(
        r".\Input data\Sample data.xlsx", sheet_name="Calibration", index_col=0
    )

    validation_data = pd.read_excel(
        r".\Input data\Sample data.xlsx", sheet_name="Validation", index_col=0
    )

    # --- CALIBRATION ---+

    # Model with calibration data
    model_cal = LWBM(
        prec=calibration_data["Prec"].values,
        temp=calibration_data["Temp"].values,
        evap=calibration_data["PET"].values,
        obs=calibration_data["Flow_o"].values,
    )
    # Objective functions setup
    objective_functions = ObjectiveFunctions(
        model=model_cal, observed_data=model_cal.obs
    )
    # Load model parameter bounds from JSON file
    bounds = load_params_bounds_json()

    # Differential evolution algorithm setup
    de_algorithm = differentital_evolution(
        obj_function=objective_functions.nse,
        popsize=210,
        bounds=bounds,
        generations=20,
        mutate=0.7,
        CR=0.8,
    )

    params = np.empty(len(bounds))
    crit = 0.0
    cnt = 0

    for cnt, res in enumerate(de_algorithm):
        params, crit = res
        if cnt % 1 == 0:
            print(f"Generation no. {cnt},\tNSE: {crit}")

    print("Calibration ended.")

    # Model setup with validation data
    model_val = LWBM(
        prec=validation_data["Prec"].values,
        temp=validation_data["Temp"].values,
        evap=validation_data["PET"].values,
        obs=validation_data["Flow_o"].values,
    )

    objective_functions = ObjectiveFunctions(
        model=model_val, observed_data=model_val.obs
    )
    # Objective function evaluation for validation data
    val_nse = objective_functions.nse(params)
    print(f"Validation objective function: {val_nse}")

    # --- EXPORT RESULTS ---+

    calibration_data["Flow_m"] = model_cal(params=params)
    validation_data["Flow_m"] = model_val(params=params)

    with pd.ExcelWriter("results.xlsx") as writer:
        calibration_data.to_excel(writer, sheet_name="Calibration", index=True)
        validation_data.to_excel(writer, sheet_name="Validation", index=True)

    # Optimised parameters export
    np.savetxt("parameters.csv", params, delimiter=",")
