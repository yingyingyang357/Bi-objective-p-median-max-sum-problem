import time
import logging
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import Optional, Tuple, ClassVar, Set
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData

# configure logger for the direct binary search module
logger = logging.getLogger(__name__)


class DirectBinarySearch(OptimisationModelBase):
    """
    Direct Binary search method for p-dispersion optimization. The Algorithm is implemented as in
    The direct binary search (DBS) algorithm from Parreno, F., Alvarez-Valdes, R., Marti, R., 2021. Measuring diversity.
    A review and an empirical analysis. European Journal of Operational Research 289, 515–532.
    """

    name: ClassVar[str] = "Direct Binary Search Method for P-Dispersion"
    bisection_ratio: float = 0.5  # bisection ratio for interpolation between bounds
    time_limit: float = 3600

    def __init__(self, **data):
        super().__init__(**data)
        self.bisection_ratio = data.get("bisection_ratio", self.bisection_ratio)
        self.time_limit = data.get("time_limit", self.time_limit)

    @staticmethod
    def find_smallest_distance_above_threshold(
        distance_array: np.ndarray, threshold: float
    ) -> Optional[float]:
        """
        Find the smallest distance in the array that is greater than or equal to the threshold.
        Uses binary search for O(log n) performance on sorted arrays.

        Args:
            distance_array: Sorted numpy array of distance values
            threshold: Minimum threshold value

        Returns:
            Smallest distance >= threshold, or None if no such distance exists
        """
        # use binary search to find insertion point for threshold
        idx = np.searchsorted(distance_array, threshold, side="left")

        # return the distance at that index if it exists
        return float(distance_array[idx]) if idx < len(distance_array) else None

    @staticmethod
    def check_dispersion_feasibility(
        num_facilities: int,
        p_median: int,
        distance_matrix: np.ndarray,
        time_limit: float,
        min_distance_threshold: float,
    ) -> bool:
        """
        Check if p-dispersion problem is feasible with given minimum distance threshold.

        Args:
            num_facilities: Total number of available facilities
            p_median: Number of facilities to select
            distance_matrix: Symmetric distance matrix (n x n)
            time_limit: Maximum solving time in seconds
            min_distance_threshold: Minimum required distance between selected facilities

        Returns:
            True if feasible (can select p facilities with min distance), False otherwise
        """
        model = Model(name="Dispersion_Feasibility_Check")
        model.set_quiet()

        # decision variables: x[i] = 1 if facility i is selected, 0 otherwise
        facility_indices = list(range(num_facilities))
        facility_selection = model.binary_var_dict(keys=facility_indices, name="x")

        # constraint: select exactly p facilities
        model.add_constraint(
            model.sum(facility_selection[i] for i in facility_indices) == p_median,
            ctname="select_p_facilities",
        )

        # conflict constraints: prevent selection of facilities that are too close
        # xi + xj <= 1, when distance[i,j] < min_distance_threshold
        model.add_constraints(
            facility_selection[i] + facility_selection[j] <= 1
            for i in range(num_facilities)
            for j in range(i + 1, num_facilities)
            if 0 < distance_matrix[i, j] < min_distance_threshold
        )

        # set solver parameters
        model.parameters.timelimit = time_limit

        # solve feasibility problem (no objective needed)
        solution = model.solve()

        return solution is not None

    def direct_binary_search_algorithm(
        self,
        problem: ProblemData,
        p_median: int,
        bisection_ratio: float,
        time_limit: float,
    ) -> Tuple[Optional[float], float, int, Set[str]]:
        """
        Direct binary search algorithm for p-dispersion optimization.

        Args:
            problem: Problem instance containing facilities and distance matrix
            p_median: Number of facilities to select
            bisection_ratio: Ratio for bisection interpolation (typically 0.5)
            time_limit: Maximum time allowed for optimization

        Returns:
            Tuple containing optimal distance, runtime, iterations, and warnings
        """
        # initialize warning collection
        warnings = set()

        # extract problem parameters
        num_facilities = problem.number_of_facilities
        distance_matrix = problem.distance_matrix.copy()
        start_time = time.time()

        # validate input parameters
        if p_median <= 0:
            warning_msg = f"Invalid p_median value: {p_median}. Must be positive."
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        if p_median > num_facilities:
            warning_msg = (
                f"p_median ({p_median}) exceeds number of facilities ({num_facilities})"
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        if not (0.0 < bisection_ratio <= 1.0):
            warning_msg = (
                f"Invalid bisection_ratio: {bisection_ratio}. Using default 0.5."
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            bisection_ratio = 0.5

        if time_limit <= 0:
            warning_msg = (
                f"Invalid time_limit: {time_limit}. Using default 3600 seconds."
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            time_limit = 3600

        # extract upper triangle distances (excluding diagonal) using numpy indexing
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        upper_triangle_distances = distance_matrix[triu_indices]

        # check for valid distance data
        if len(upper_triangle_distances) == 0:
            warning_msg = "No distance data available in the upper triangle matrix"
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        # efficiently extract unique distances using numpy
        unique_distances = np.unique(upper_triangle_distances)

        # check for sufficient distance variation
        if len(unique_distances) <= 1:
            warning_msg = f"Insufficient distance variation: only {len(unique_distances)} unique distance(s) found"
            warnings.add(warning_msg)
            logger.warning(warning_msg)

        # find minimum resolution using vectorized numpy operations
        if len(unique_distances) > 1:
            distance_diffs = np.diff(unique_distances)
            non_zero_diffs = distance_diffs[distance_diffs > 0]
            min_resolution = (
                float(np.min(non_zero_diffs))
                if len(non_zero_diffs) > 0
                else float("inf")
            )
        else:
            min_resolution = float("inf")

        # initialize bisection search bounds
        lower_bound = unique_distances[0]
        upper_bound = unique_distances[-1]
        iteration_count = 0
        runtime = 0

        # main bisection loop
        while upper_bound - lower_bound > min_resolution and runtime <= time_limit:
            # calculate test point using bisection ratio
            test_distance = lower_bound + (upper_bound - lower_bound) * bisection_ratio

            # check feasibility at test distance
            is_feasible = self.check_dispersion_feasibility(
                num_facilities, p_median, distance_matrix, time_limit, test_distance
            )

            # update bounds based on feasibility
            if not is_feasible:
                upper_bound = test_distance
            else:
                lower_bound = test_distance

            # check for convergence using efficient binary search
            # find unique distances in current search interval [lower_bound, upper_bound)
            left_idx = np.searchsorted(unique_distances, lower_bound, side="left")
            right_idx = np.searchsorted(unique_distances, upper_bound, side="left")
            distances_in_interval = unique_distances[left_idx:right_idx]

            # convergence check: if exactly one unique distance remains, we found the optimum
            if len(distances_in_interval) == 1:
                optimal_distance = float(distances_in_interval[0])
                runtime = time.time() - start_time
                logger.info(
                    f"Direct binary search converged after {iteration_count} iterations to distance {optimal_distance:.6f}"
                )
                return optimal_distance, runtime, iteration_count, warnings

            # update iteration tracking
            iteration_count += 1
            runtime = time.time() - start_time

            # check for time limit approached
            if runtime >= time_limit * 0.95:  # 95% of time limit
                warning_msg = f"Approaching time limit ({runtime:.2f}s / {time_limit}s). Search may be terminated early."
                warnings.add(warning_msg)
                logger.warning(warning_msg)

        # fallback: find closest feasible distance if no exact convergence
        optimal_distance = self.find_smallest_distance_above_threshold(
            unique_distances, lower_bound
        )

        # log completion information
        if optimal_distance is not None:
            warning_msg = (
                "Direct binary search completed without exact convergence. "
                f"Returning closest feasible distance: {optimal_distance:.6f}"
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
        else:
            warning_msg = "Direct binary search failed to find any feasible distance"
            warnings.add(warning_msg)
            logger.error(warning_msg)

        return optimal_distance, runtime, iteration_count, warnings

    def optimise(
        self, problem: ProblemData, p_median: int, validation: bool = False
    ) -> Tuple[Set[str], pd.DataFrame]:
        """
        The main optimise function of the Direct Binary Search method for p-dispersion problems.

        Args:
            problem (ProblemData): The problem data containing distance matrix and number of facilities.
            p_median (int): The number of facilities to be selected.
            validation (bool): Whether to perform input validation before optimization.
        Returns:
            A tuple containing:
                Set[str]: A set of warning messages produced by the optimisation model
                pd.DataFrame: A dataframe containing an optimised schedule
        """
        # initialize warning collection for this optimization run
        optimization_warnings = set()

        # validate input parameters before optimization
        if validation:
            if problem.number_of_facilities <= 0:
                warning_msg = (
                    f"Invalid number of facilities: {problem.number_of_facilities}"
                )
                optimization_warnings.add(warning_msg)
                logger.error(warning_msg)
                return optimization_warnings, pd.DataFrame()

        if problem.distance_matrix is None:
            warning_msg = "Distance matrix is None. Cannot perform optimization."
            optimization_warnings.add(warning_msg)
            logger.error(warning_msg)
            return optimization_warnings, pd.DataFrame()

        if self.time_limit <= 0:
            warning_msg = (
                f"Invalid time limit: {self.time_limit}. Using default 3600 seconds."
            )
            optimization_warnings.add(warning_msg)
            logger.warning(warning_msg)
            self.time_limit = 3600

        # log optimization start
        logger.info(
            f"Starting direct binary search optimization for {problem.number_of_facilities} facilities, p={p_median}"
        )

        try:
            # run direct binary search algorithm
            optimal_distance, runtime, iterations, search_warnings = (
                self.direct_binary_search_algorithm(
                    problem=problem,
                    p_median=p_median,
                    bisection_ratio=self.bisection_ratio,
                    time_limit=self.time_limit,
                )
            )

            # combine warnings from search with optimization warnings
            optimization_warnings.update(search_warnings)

            # create results DataFrame
            if optimal_distance is not None:
                results_df = pd.DataFrame(
                    {
                        "algorithm": [self.name],
                        "p_median": [p_median],
                        "optimal_distance": [optimal_distance],
                        "runtime_seconds": [runtime],
                        "iterations": [iterations],
                        "ratio": [self.bisection_ratio],
                        "status": [
                            "optimal" if len(search_warnings) == 0 else "suboptimal"
                        ],
                    }
                )

                logger.info(
                    f"Direct binary search completed: distance={optimal_distance:.6f}, "
                    f"runtime={runtime:.3f}s, iterations={iterations}"
                )
            else:
                results_df = pd.DataFrame(
                    {
                        "algorithm": [self.name],
                        "p_median": [p_median],
                        "optimal_distance": [None],
                        "runtime_seconds": [runtime],
                        "iterations": [iterations],
                        "ratio": [self.bisection_ratio],
                        "status": ["failed"],
                    }
                )
                warning_msg = (
                    f"Direct binary search failed to find solution for p={p_median}"
                )
                optimization_warnings.add(warning_msg)
                logger.error(warning_msg)

        except Exception as e:
            error_msg = (
                f"Direct binary search optimization failed with exception: {str(e)}"
            )
            optimization_warnings.add(error_msg)
            logger.error(error_msg, exc_info=True)

            # return empty results on exception
            results_df = pd.DataFrame(
                {
                    "algorithm": [self.name],
                    "p_median": [p_median],
                    "optimal_distance": [None],
                    "runtime_seconds": [0.0],
                    "iterations": [0],
                    "ratio": [self.bisection_ratio],
                    "status": ["error"],
                }
            )

        # log final warning count
        if len(optimization_warnings) > 0:
            logger.warning(
                f"Optimization completed with {len(optimization_warnings)} warning(s)"
            )

        return optimization_warnings, results_df
