from __future__ import annotations
from typing import Callable
import numpy as np
import numpy.typing as npt


def differentital_evolution(
    obj_function: Callable[[npt.NDArray, npt.NDArray], float],
    popsize: int,
    bounds: list[tuple[float, float]],
    generations: int,
    mutate: float,
    CR: float,
):
    """
    Differential evolution algorithm using the "DE/rand/1/bin" variant.

    Args:
        obj_function (Callable[[npt.NDArray, npt.NDArray], float]): Any performance measure function
        popsize (int): Size of population (it is recommended to be at least 2-10 times as number of model parameters)
        bounds (list[tuple[float, float]]): List of tuples representing lower and upper bounds of model parameters
        generations (int): Number of generations
        mutate (float): Mutation constant
        CR (float): Crossover threshold

    Yields:
        _type_: _description_
    """
    dimension = len(bounds)

    population_norm: npt.NDArray[np.float64] = np.random.rand(popsize, dimension)

    min_bound, max_bound = np.asarray(bounds).T

    difference = np.fabs(min_bound - max_bound)

    population = min_bound + population_norm * difference

    fitness = np.asarray([obj_function(parameters) for parameters in population])

    best_idx = np.argmin(fitness)

    best = population[best_idx]

    for _ in range(generations + 1):
        for j in range(popsize):
            idxs_0 = np.asarray(range(0, popsize))
            idxs = np.delete(idxs_0, j)

            x_1, x_2, x_3 = population_norm[np.random.choice(idxs, 3, replace=True)]

            noise_vector = np.clip(x_1 + mutate * (x_2 - x_3), 0, 1)

            cross_points = np.random.rand(dimension) < CR

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimension)] = True

            trial_vector_norm = np.where(cross_points, noise_vector, population_norm[j])
            trial_vector = min_bound + trial_vector_norm * difference
            criterium = obj_function(trial_vector)

            if criterium < fitness[j]:
                fitness[j] = criterium
                population_norm[j] = trial_vector_norm

                if criterium < fitness[best_idx]:
                    best_idx = j
                    best = trial_vector

        yield best, 1 - fitness[
            best_idx
        ]  # Yield of fitness values are set for NSE (see above), thus "1 - fitness"
