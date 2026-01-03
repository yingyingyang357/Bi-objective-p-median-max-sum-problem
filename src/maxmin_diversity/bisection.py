import time
import logging
from itertools import combinations
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import Optional, Tuple, ClassVar, Set
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData

# configure logger for the bisection module
logger = logging.getLogger(__name__)


class BisectionMethod(OptimisationModelBase):
    """
    Bisection method for p-dispersion optimization in bi-objective p-median problems.
    The Algorithm is implemented as in
    Yingying Yang, Hoa T. Bui, Ryan Loxton. An Exact Method for the Bi-objective p-median Max-sum Diversity Problem.

    Uses adaptive bisection search to find maximum dispersion distance for facility selection.
    """

    name: ClassVar[str] = "Adaptive Bisection Method for P-Dispersion"
    ratio: float = 0.5  # bisection ratio for interpolation between bounds
    time_limit: float = 3600

    def __init__(self, **data):
        super().__init__(**data)
        self.ratio = data.get("ratio", self.ratio)
        self.time_limit = data.get("time_limit", self.time_limit)

    @staticmethod
    def check_dispersion_feasibility(
        n: int,
        p: int,
        distance_matrix: np.ndarray,
        min_distance: float,
        time_limit: float,
    ) -> Optional[float]:
        """
        Check feasibility of p-dispersion problem with minimum distance threshold.

        Solves the feasibility problem: Can we select p facilities such that
        the minimum pairwise distance is at least min_distance?

        Args:
            n: Number of facilities
            p: Number of facilities to select
            distance_matrix: Symmetric distance matrix (n x n)
            min_distance: Minimum distance threshold to test
            time_limit: Maximum solving time in seconds

        Returns:
            float: Actual minimum distance achieved if feasible
            None: If problem is infeasible
        """
        # build optimisation model for feasibility checking
        model = Model(name=f"p_dispersion_feasibility_mdist_{min_distance:.3f}")
        model.set_quiet()

        # binary decision variables: x[i] = 1 if facility i is selected
        x = model.binary_var_list(n, name="select")

        # constraint: select exactly p facilities
        model.add_constraint(model.sum(x) == p, ctname="select_p_facilities")

        # dispersion constraints: prevent close facilities from being selected together
        # for each pair (i,j) with distance < min_distance, at most one can be selected
        conflict_pairs = {
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if 0 < distance_matrix[i, j] < min_distance
        }

        if conflict_pairs:
            model.add_constraints(
                (x[i] + x[j] <= 1, f"dispersion_{i}_{j}") for i, j in conflict_pairs
            )

        # configure solver parameters
        model.parameters.timelimit = time_limit
        model.parameters.mip.tolerances.mipgap = 0.0

        # solve feasibility problem (set any objective value)
        model.minimize(0)
        solution = model.solve()

        # process solution after solving
        if solution is not None:
            # get selected facilities
            selected_facilities = [
                i for i, var in enumerate(x) if solution.get_value(var) > 0.5
            ]

            # calculate actual minimum distance achieved
            if len(selected_facilities) >= 2:
                pairwise_distances = [
                    distance_matrix[i, j]
                    for i, j in combinations(selected_facilities, 2)
                    if distance_matrix[i, j] > 0
                ]
                actual_min_distance = (
                    min(pairwise_distances) if pairwise_distances else float("inf")
                )
                return actual_min_distance
            else:
                return float("inf")  # single facility case

        return None  # infeasible

    def bi_section(
        self, problem: ProblemData, p_median: int, ratio: float
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
            ratio: Bisection ratio for interpolation between bounds (typically 0.5 for midpoint)

        Returns:
            Tuple containing:
            - float or None: Maximum achievable minimum dispersion distance, None if no solution
            - float: Total runtime in seconds
            - int: Number of bisection iterations performed
        """
        # initialize warning collection
        warnings = set()

        # extract problem parameters
        n = problem.number_of_facilities
        distance_matrix = problem.distance_matrix.copy()
        start_time = time.time()

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

        if not (0.0 < ratio <= 1.0):
            warning_msg = f"Invalid ratio: {ratio}. Using default 0.5."
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            ratio = 0.5

        if self.time_limit <= 0:
            warning_msg = (
                f"Invalid time_limit: {self.time_limit}. Using default 3600 seconds."
            )
            warnings.add(warning_msg)
            logger.warning(warning_msg)
            self.time_limit = 3600

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
        sorted_distances = np.sort(upper_triangle_distances)
        max_removable = max(0, round(p_median * (p_median - 1) * 0.5) - 1)

        # calculate feasible distance range endpoint
        end_idx = (
            len(sorted_distances) - max_removable
            if max_removable > 0
            else len(sorted_distances)
        )
        end_idx = max(0, min(end_idx, len(sorted_distances)))

        # extract unique feasible distances
        unique_distances = np.unique(sorted_distances[:end_idx])

        # check for sufficient distance variation
        if len(unique_distances) <= 1:
            warning_msg = f"Insufficient distance variation: only {len(unique_distances)} unique distance(s) found"
            warnings.add(warning_msg)
            logger.warning(warning_msg)

        # find smallest non-zero difference between consecutive unique distances
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

        # handle edge case: no feasible distances
        if len(unique_distances) == 0:
            warning_msg = "No feasible distances found for p-dispersion optimization"
            warnings.add(warning_msg)
            logger.error(warning_msg)
            runtime = time.time() - start_time
            return None, runtime, 0

        # initialize bisection bounds with min/max feasible distances
        lower_bound = float(unique_distances[0])
        upper_bound = float(unique_distances[-1])
        iteration_count = 0

        # main bisection loop with convergence and time limit checks
        while (
            upper_bound - lower_bound > min_resolution
            and (time.time() - start_time) <= self.time_limit
        ):

            # calculate interpolation point using specified ratio
            test_distance = lower_bound + (upper_bound - lower_bound) * ratio

            # check feasibility of test distance using optimization solver
            feasibility_result = self.check_dispersion_feasibility(
                n=n,
                p=p_median,
                distance_matrix=distance_matrix,
                time_limit=self.time_limit,
                min_distance=test_distance,
            )

            # update bounds based on feasibility result
            if feasibility_result is not None:
                # Feasible: increase lower bound, potentially use better achieved distance
                lower_bound = max(test_distance, feasibility_result)
            else:
                # Infeasible: decrease upper bound
                upper_bound = test_distance

            # check for convergence to single distance value
            # find distances in current search interval [lower_bound, upper_bound)
            left_idx = np.searchsorted(unique_distances, lower_bound, side="left")
            right_idx = np.searchsorted(unique_distances, upper_bound, side="left")

            interval_distances = unique_distances[left_idx:right_idx]

            if len(interval_distances) == 1:
                # converged to single distance value
                runtime = time.time() - start_time
                logger.info(
                    f"Bisection converged after {iteration_count} iterations to distance {float(interval_distances[0]):.6f}"
                )
                return float(interval_distances[0]), runtime, iteration_count, warnings

            iteration_count += 1

            # check for time limit approached
            current_runtime = time.time() - start_time
            if current_runtime >= self.time_limit * 0.95:  # 95% of time limit
                warning_msg = f"Approaching time limit ({current_runtime:.2f}s / {self.time_limit}s). Search may be terminated early."
                warnings.add(warning_msg)
                logger.warning(warning_msg)

        # find closest feasible distance when loop terminates
        if len(unique_distances) > 0:
            closest_idx = np.searchsorted(unique_distances, lower_bound, side="left")
            optimal_distance = (
                float(unique_distances[closest_idx])
                if closest_idx < len(unique_distances)
                else None
            )
        else:
            optimal_distance = None

        runtime = time.time() - start_time

        # log completion information
        if optimal_distance is not None:
            warning_msg = f"Bisection completed without exact convergence. Returning closest feasible distance: {optimal_distance:.6f}"
            warnings.add(warning_msg)
            logger.warning(warning_msg)
        else:
            warning_msg = "Bisection search failed to find any feasible distance"
            warnings.add(warning_msg)
            logger.error(warning_msg)

        return optimal_distance, runtime, iteration_count, warnings

    def optimise(
        self, problem: ProblemData, p_median: int, validation: bool = False
    ) -> Tuple[Set[str], pd.DataFrame]:
        """
        Optimise using the bisection method.

        Args:
            problem (ProblemData): The problem data containing facilities and distance matrix.
            p_median (int): Number of facilities to select for p-dispersion.
            validation (bool): Whether to perform additional validation checks.

        Returns:
            Tuple[Set[str], pd.DataFrame]: Collection of warnings and optimization results.
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
            f"Starting bisection optimization for {problem.number_of_facilities} facilities, p={p_median}"
        )

        try:
            # run bisection algorithm
            optimal_distance, runtime, iterations, search_warnings = self.bi_section(
                problem=problem, p_median=p_median, ratio=self.ratio
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
                        "ratio": [self.ratio],
                        "status": [
                            "optimal" if len(search_warnings) == 0 else "suboptimal"
                        ],
                    }
                )

                logger.info(
                    f"Bisection completed: distance={optimal_distance:.6f}, "
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
                        "ratio": [self.ratio],
                        "status": ["failed"],
                    }
                )

                warning_msg = f"Bisection failed to find solution for p={p_median}"
                optimization_warnings.add(warning_msg)
                logger.error(warning_msg)

        except Exception as e:
            error_msg = f"Bisection optimization failed with exception: {str(e)}"
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
                    "ratio": [self.ratio],
                    "status": ["error"],
                }
            )

        # log final warning count
        if len(optimization_warnings) > 0:
            logger.warning(
                f"Optimization completed with {len(optimization_warnings)} warning(s)"
            )

        return optimization_warnings, results_df
