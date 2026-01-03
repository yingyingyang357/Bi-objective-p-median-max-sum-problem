import time
import math
import logging
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import List, Optional, Tuple, ClassVar, Set
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData

# configure logger for the binary search module
logger = logging.getLogger(__name__)


class BinarySearch(OptimisationModelBase):
    """
    Binary search method for p-dispersion optimization. The Algorithm is implemented as in
    Sayyady, F., Fathi, Y., 2016. An integer programming approach for solving the p-dispersion problem.
    European Journal of Operational Research 253, 216–225.
    """

    name: ClassVar[str] = "Binary Search Method for P-Dispersion"
    ratio: float = 0.5  # bisection ratio for interpolation between bounds
    time_limit: float = 3600

    def __init__(self, **data):
        super().__init__(**data)
        self.ratio = data.get("ratio", self.ratio)
        self.time_limit = data.get("time_limit", self.time_limit)

    # find the solution for mdis: how many and which nodes are chosen. For the Sayyady and Fathi (2016) paper
    @staticmethod
    def solve_max_dispersion_feasibility(
        num_facilities: int,
        distance_matrix: np.ndarray,
        time_limit: float,
        min_distance_threshold: float,
    ) -> Tuple[List[int], int]:
        """
        Solve the maximum dispersion feasibility problem for p-facility selection.

        This function implements the feasibility check from Sayyady & Fathi (2016) for the
        p-dispersion problem. It finds the maximum number of facilities that can be selected
        such that all pairwise distances are at least min_distance_threshold.

        Mathematical formulation:
            maximize: Σ x_i  (total number of selected facilities)
            subject to: x_i + x_j ≤ 1, ∀(i,j) where 0 < d_ij < min_distance_threshold
                    x_i ∈ {0,1}, ∀i

        Args:
            num_facilities: Total number of available facilities (n)
            distance_matrix: Symmetric distance matrix (n x n numpy array)
            time_limit: Maximum solving time in seconds
            min_distance_threshold: Minimum required pairwise distance between selected facilities

        Returns:
            Tuple containing:
            - List[int]: Indices of selected facilities in the optimal solution
            - int: Maximum number of facilities that can be selected (objective value)

        Note:
            If the returned objective value >= p_median, then the dispersion constraint
            is feasible for p facilities at the given distance threshold.
        """
        # create optimization model for maximum dispersion problem
        model = Model(name="Maximum_Dispersion_Feasibility")

        # decision variables: x[i] = 1 if facility i is selected, 0 otherwise
        facility_indices = list(range(num_facilities))
        facility_selection = model.binary_var_dict(keys=facility_indices, name="x")

        # conflict constraints: prevent selection of facilities that are too close
        # add constraint x[i] + x[j] ≤ 1 when distance d[i,j] < min_distance_threshold
        model.add_constraints(
            facility_selection[i] + facility_selection[j] <= 1
            for i in range(num_facilities)
            for j in range(i + 1, num_facilities)
            if 0 < distance_matrix[i, j] < min_distance_threshold
        )

        # objective: maximize the total number of selected facilities
        model.maximize(model.sum(facility_selection[i] for i in facility_indices))

        # set solver parameters
        model.parameters.timelimit = time_limit
        model.set_quiet()  # suppress solver output

        # solve the optimization problem
        solution = model.solve()

        # extract solution information
        if solution is not None:
            max_facilities_achievable = int(model.objective_value)

            # identify selected facilities from the solution
            selected_facilities = []
            for i in facility_indices:
                if solution.get_value(facility_selection[i]) > 0.5:  # binary threshold
                    selected_facilities.append(i)

            return selected_facilities, max_facilities_achievable
        else:
            # no solution found (should not happen for this formulation)
            return [], 0

    # Binary_Search from the Sayyady and Fathi (2016) paper
    def binary_search(
        self, problem: ProblemData, p_median: int
    ) -> Tuple[Optional[float], float, int, Set[str]]:
        """
        Bisection search algorithm for finding minimum maximum p-dispersion distance

        Uses a specialized bisection method that leverages the discrete nature of distance values
        to efficiently find the optimal minimum dispersion distance for p-facility selection.
        The algorithm filters distances to remove those impossible in feasible solutions and
        uses adaptive bisection with feasibility checking.

        Args:
            problem: Problem instance containing facilities, distance matrix, and time limits
            p_median: Number of facilities to select (p-dispersion parameter)

        Returns:
            Tuple containing:
            - float or None: Maximum achievable minimum dispersion distance, None if no solution
            - float: Total runtime in seconds
            - int: Number of bisection iterations performed
            - Set[str]: Collection of warning messages encountered during optimization
        """
        # initialize warning collection
        warnings = set()

        # extract problem parameters
        n = problem.number_of_facilities
        distance_matrix = problem.distance_matrix.copy()
        start_time = time.time()
        time_limit = self.time_limit

        # validate input parameters
        if p_median <= 0:
            warning_msg = f"Invalid p_median value: {p_median}. Must be positive."
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        if p_median > n:
            warning_msg = f"p_median ({p_median}) exceeds number of facilities ({n})"
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        if time_limit <= 0:
            warning_msg = (
                f"Invalid time_limit: {time_limit}. Using default 3600 seconds."
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            time_limit = 3600

        # indexing for upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        upper_triangle_distances = distance_matrix[triu_indices]

        # check for valid distance data
        if len(upper_triangle_distances) == 0:
            warning_msg = "No distance data available in the upper triangle matrix"
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            return None, time.time() - start_time, 0, warnings

        # sort distances and filter out impossible values for p-dispersion
        # remove largest (p*(p-1)/2 - 1) distances since they cannot be achieved
        # in any feasible p-facility solution due to geometric constraints
        # extract unique feasible distances
        unique_distances = np.unique(upper_triangle_distances)
        sorted_distances = np.sort(unique_distances)

        # check for sufficient distance variation
        if len(unique_distances) <= 1:
            warning_msg = f"Insufficient distance variation: only {len(unique_distances)} unique distance(s) found"
            warnings.add(warning_msg)
            logger.warning(warning_msg)

        # find smallest non-zero difference between consecutive unique distances
        if len(sorted_distances) > 1:
            distance_diffs = np.diff(sorted_distances)
            non_zero_diffs = distance_diffs[distance_diffs > 0]
            min_resolution = (
                float(np.min(non_zero_diffs))
                if len(non_zero_diffs) > 0
                else float("inf")
            )
        else:
            min_resolution = float("inf")

        # add the last element plus min_resolution to the list
        add_last = sorted_distances[-1] + min_resolution
        sorted_distances = np.append(sorted_distances, add_last)

        # find the smallest positive integer such that 2^(-grid_power)*(distance_range) <= min_resolution
        grid_power = (
            0
            if min_resolution == float("inf")
            else int(
                math.ceil(
                    math.log2(
                        (sorted_distances[-1] - sorted_distances[0]) / min_resolution
                    )
                )
            )
        )
        step_size = (sorted_distances[-1] - sorted_distances[0]) / (2**grid_power)

        # generate a uniform grid with 2^grid_power + 1 elements with equal step_size intervals
        grid_distances = [
            sorted_distances[0] + step_size * i for i in range(2**grid_power + 1)
        ]

        # initialize binary search with 1-based indexing for the grid
        iteration_count = 0
        lower_bound = 1
        upper_bound = 2**grid_power + 1
        optimal_distance = None
        runtime = 0.0

        # main binary search loop: find the maximum feasible dispersion distance
        for i in range(1, grid_power + 1):
            # calculate midpoint using binary search formula from Sayyady & Fathi (2016)
            midpoint = int(lower_bound + 2 ** (grid_power - i))
            test_distance = grid_distances[midpoint - 1]

            # solve feasibility problem for the test distance threshold
            _, max_facilities = self.solve_max_dispersion_feasibility(
                num_facilities=n,
                distance_matrix=distance_matrix,
                time_limit=time_limit,
                min_distance_threshold=test_distance,
            )

            # update search bounds based on feasibility result
            if max_facilities >= p_median:
                # feasible: can select at least p_median facilities, search for larger distances
                lower_bound = midpoint
            else:
                # infeasible: cannot select p_median facilities, search for smaller distances
                upper_bound = midpoint

            # check convergence: find unique distances in current search interval
            # convert grid indices back to actual distance values
            lower_distance = grid_distances[lower_bound - 1]
            upper_distance = grid_distances[upper_bound - 1]

            # use binary search to efficiently count distances in interval [lower_distance, upper_distance)
            left_idx = np.searchsorted(unique_distances, lower_distance, side="left")
            right_idx = np.searchsorted(unique_distances, upper_distance, side="left")
            distances_in_interval = unique_distances[left_idx:right_idx]

            # convergence check: if exactly one unique distance remains, we found the optimum
            if len(distances_in_interval) == 1:
                optimal_distance = float(distances_in_interval[0])
                runtime = time.time() - start_time
                logger.info(
                    f"Binary search converged after {iteration_count} iterations to distance {optimal_distance:.6f}"
                )
                return optimal_distance, runtime, iteration_count, warnings

            # update iteration count and runtime
            iteration_count += 1
            runtime = time.time() - start_time

            # check for time limit exceeded
            if runtime >= time_limit:
                warning_msg = f"Approaching time limit ({runtime:.2f}s / {time_limit}s). Search may be terminated early."
                warnings.add(warning_msg)
                logger.warning(warning_msg)

        # find the closest feasible distance if loop completes without convergence
        final_distance = (
            grid_distances[lower_bound - 1]
            if lower_bound <= len(grid_distances)
            else None
        )
        closest_idx = np.searchsorted(unique_distances, final_distance, side="left")
        optimal_distance = (
            float(unique_distances[closest_idx])
            if closest_idx < len(unique_distances)
            else None
        )

        # log convergence information
        if optimal_distance is not None:
            warning_msg = (
                "Binary search completed without exact convergence. "
                + f"Returning closest feasible distance: {optimal_distance:.6f}"
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
        else:
            warning_msg = "Binary search failed to find any feasible distance"
            warnings.add(warning_msg)
            logger.error(warning_msg)

        return optimal_distance, runtime, iteration_count, warnings

    def optimise(
        self, problem: ProblemData, p_median: int, validation: bool = False
    ) -> Tuple[Set[str], pd.DataFrame]:
        """
        Optimise using the binary search method.

        Args:
            problem (ProblemData): The problem data containing facilities and distance matrix.
            p_median (int): Number of facilities to select for p-dispersion.
            validation (bool): Whether to perform additional validation checks.

        Returns:
            Tuple[Set[str], pd.DataFrame]: Collection of warnings and optimization results.
        """
        # initialize warning collection for this optimization run
        optimization_warnings = set()

        # If validation is enabled, perform input checks
        if validation:
            # validate input parameters before optimization
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
                warning_msg = f"Invalid time limit: {self.time_limit}. Using default 3600 seconds."
                optimization_warnings.add(warning_msg)
                logger.warning(warning_msg)
                self.time_limit = 3600

        # log optimization start
        logger.info(
            f"Starting binary search optimization for {problem.number_of_facilities} facilities, p={p_median}"
        )

        try:
            # run binary search algorithm
            optimal_distance, runtime, iterations, search_warnings = self.binary_search(
                problem=problem, p_median=p_median
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
                        "status": [
                            "optimal" if len(search_warnings) == 0 else "suboptimal"
                        ],
                    }
                )

                logger.info(
                    f"Binary search completed: distance={optimal_distance:.6f}, "
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
                        "status": ["failed"],
                    }
                )

                warning_msg = f"Binary search failed to find solution for p={p_median}"
                optimization_warnings.add(warning_msg)
                logger.error(warning_msg)

        except Exception as e:
            error_msg = f"Binary search optimization failed with exception: {str(e)}"
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
                    "status": ["error"],
                }
            )

        # log final warning count
        if len(optimization_warnings) > 0:
            logger.warning(
                f"Optimization completed with {len(optimization_warnings)} warning(s)"
            )

        return optimization_warnings, results_df
