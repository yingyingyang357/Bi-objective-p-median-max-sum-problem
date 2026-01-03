import time
import logging
import numpy as np
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from docplex.mp.constr import LinearConstraint, QuadraticConstraint
from typing import List, Optional, Tuple, ClassVar, Dict, Any
from src.data_class import ProblemData

# configure logger for the beta model module
logger = logging.getLogger(__name__)


class BetaPmeanModel:
    """
    Beta-constraint model for bi-objective p-median optimization with dispersion requirements.

    This class implements a mathematical programming formulation that solves the bi-objective
    p-median problem by constraining one objective (dispersion) and optimizing the other
    (median cost) using the epsilon-constraint method, also known as the beta-constraint method.

    The model addresses the following optimization problem:
    - Primary objective: Minimize total assignment cost (p-median objective)
    - Constraint: Achieve minimum dispersion (sum of squared distances between selected facilities) ≥ β

    Mathematical Formulation:
    - Decision variables: x_i ∈ {0,1} (facility selection), y_ik ∈ {0,1} (assignment)
    - Minimize: Σ d_ik * y_ik (total assignment cost)
    - Subject to: Σ d_ij * x_i * x_j ≥ β + ε (dispersion constraint)
    - Plus standard p-median constraints (coverage, capacity, conflict avoidance)

    Methods:
    - quadratic_model(): Solves using direct quadratic programming (exact but potentially slow)
    - cutting_plane_model(): Solves using cutting plane approximation (faster for large problems)

    Attributes:
        p_median (int): Number of facilities to select
        min_distance (float): Minimum required distance between any two selected facilities
        problem_data (ProblemData): Instance data containing facilities and distance matrix
        validate (bool): Whether to perform input validation and detailed logging
        time_limit (int): Maximum solving time in seconds (default: 3600)

    Note:
        This implementation follows the beta-constraint approach for multi-objective optimization,
        where the dispersion objective is converted to a constraint with parameter β (beta).
        Higher β values require more dispersed facility selections but may become infeasible.
    """

    name: ClassVar[str] = "P-Median Beta Model Solver"
    time_limit: int = 3600

    def __init__(
        self,
        p_median: int,
        min_distance: float,
        problem_data: ProblemData,
        validate: bool = False,
        **data,
    ) -> None:
        """
        Initialize the Beta P-Mean Model for p-median optimization.

        Args:
            p_median: Number of facilities to select as medians
            min_distance: Minimum required distance between selected facilities
            problem_data: Problem instance containing facilities and distance matrix
            **data: Additional configuration parameters
        """
        super().__init__(**data)

        # validate input parameters first
        if validate:
            if problem_data is None:
                error_msg = "Problem data cannot be None."
                logger.error(error_msg)
                raise ValueError(error_msg)

            if p_median <= 0:
                error_msg = f"Invalid p_median value: {p_median}. Must be positive."
                logger.error(error_msg)
                raise ValueError(error_msg)

            if min_distance < 0:
                error_msg = (
                    f"Invalid min_distance value: {min_distance}. Must be non-negative."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # store all instance attributes immediately after validation
        self.problem_data = problem_data
        self.p_median = p_median
        self.min_distance = min_distance
        self.validate = validate
        self.time_limit = data.get("time_limit", self.time_limit)
        self.number_of_locations = problem_data.number_of_facilities
        self.name = "p-Median with Beta Model"

        if validate:
            # validate p_median against number of facilities
            if self.p_median > self.number_of_locations:
                error_msg = f"p_median ({self.p_median}) exceeds number of facilities ({self.number_of_locations})"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if self.time_limit <= 0:
                warning_msg = f"Invalid time_limit: {self.time_limit}. Using default 3600 seconds."
                logger.warning(warning_msg)
                self.time_limit = 3600

            logger.info(
                f"Initializing Beta Quadratic Model: p={self.p_median}, "
                f"facilities={self.number_of_locations}, min_distance={self.min_distance:.4f}"
            )

        # optimization model instance
        self.p_median_model: Model = Model(name="BQP model")

        # decision variable indices
        self.yik: List[Tuple[int, int]] = [
            (i, k)
            for i in range(self.number_of_locations)
            for k in range(self.number_of_locations)
        ]
        self.xi: List[int] = [i for i in range(self.number_of_locations)]

        # decision variables with type annotations
        self.y: Dict[Tuple[int, int], Var] = self.p_median_model.binary_var_dict(
            keys=self.yik, name="y"
        )
        self.x: List[Var] = self.p_median_model.binary_var_list(keys=self.xi, name="x")

        logger.debug(
            f"Created {len(self.yik)} assignment variables and {len(self.xi)} facility variables"
        )

        # register the constraints that do not depend on beta
        # single assignment
        self.p_median_model.add_constraints(
            self.p_median_model.sum(self.y[i, k] for i in self.xi) == 1
            for k in range(self.number_of_locations)
        )
        # p-center
        self.p_median_model.add_constraint(
            self.p_median_model.sum(self.x[i] for i in self.xi) == self.p_median
        )
        # only assign to center
        self.p_median_model.add_constraints(
            self.y[i, k] <= self.x[i] for i, k in self.yik
        )
        # xi+xj<=1, when dij<m (conflict constraints for close facilities)
        self.conflict_pairs = [
            (i, j)
            for i in range(self.number_of_locations)
            for j in range(i + 1, self.number_of_locations)
            if 0 < self.problem_data.distance_matrix[i, j] < self.min_distance
        ]

        if self.conflict_pairs:
            self.p_median_model.add_constraints(
                self.x[i] + self.x[j] <= 1 for i, j in self.conflict_pairs
            )
            logger.info(
                f"Added {len(self.conflict_pairs)} conflict constraints for facilities closer than {self.min_distance:.4f}"
            )
        else:
            logger.info(
                "No conflict constraints added - all facilities satisfy minimum distance requirement"
            )
        # set objective function
        self.p_median_model.minimize(
            self.p_median_model.sum(
                self.problem_data.distance_matrix[i, k] * self.y[i, k]
                for i, k in self.yik
            )
        )
        logger.info(
            f"Beta Quadratic Model initialized successfully with {self.p_median_model.number_of_constraints} constraints"
        )

    @staticmethod
    def calculate_quadratic_objective(
        facility_selection: np.ndarray, distance_matrix: np.ndarray
    ) -> float:
        """
        Calculate the quadratic objective function value for facility selection.

        This function computes the sum of squared distances for the selected facilities:
        f(x) = x^T * Q * x

        Where:
        - x is the binary facility selection vector
        - Q is the distance matrix
        - The result represents the total dispersion (sum of pairwise distances)

        Args:
            facility_selection: Binary numpy array indicating which facilities are selected
            distance_matrix: Symmetric distance matrix (n x n numpy array)

        Returns:
            float: The quadratic objective value (sum of squared distances between selected facilities)

        Note:
            This is used in the beta-constraint method to evaluate dispersion constraints
            and in cutting plane algorithms to compute objective function values.
        """
        # create a copy to avoid modifying the original array
        selection_copy: np.ndarray = facility_selection.copy()

        # calculate quadratic form: x^T * Q * x
        return float(selection_copy.dot(distance_matrix).dot(selection_copy))

    @staticmethod
    def calculate_quadratic_gradient(
        facility_selection: np.ndarray, distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the gradient of the quadratic objective function.

        This function computes the gradient of f(x) = x^T * Q * x with respect to x:
        ∇f(x) = 2 * Q * x

        However, for symmetric matrices Q (which distance matrices are), this simplifies to:
        ∇f(x) = Q * x (since Q = Q^T for symmetric matrices)

        Args:
            facility_selection: Binary numpy array indicating current facility selection
            distance_matrix: Symmetric distance matrix (n x n numpy array)

        Returns:
            np.ndarray: Gradient vector of the quadratic function at the current point

        Note:
            This gradient is used in cutting plane methods to generate linear approximations
            of the quadratic dispersion constraint. Each component represents the marginal
            change in dispersion when facility i is selected/deselected.
        """
        # create a copy to avoid modifying the original array
        selection_copy: np.ndarray = facility_selection.copy()

        # calculate gradient: Q * x (for symmetric Q, this equals 2*Q*x/2 = Q*x)
        gradient: np.ndarray = distance_matrix.dot(selection_copy)

        return gradient

    def quadratic_model(
        self,
        beta: float,
        epsilon: float = 1,
        current_time: float = 0.0,
    ) -> Tuple[np.ndarray, float, float, float, str]:
        """
        Solve the p-median problem with beta-constraint for minimum dispersion.

        Args:
            beta: Minimum required dispersion value (sum of squared distances)
            epsilon: Small tolerance value to ensure strict inequality

        Returns:
            Union of:
            - Tuple[np.ndarray, float, float, float]: (solution_vector, objective_value, runtime, achieved_beta)
            - str: "infeasible" if no solution exists
        """
        # validate input parameters
        if self.validate:
            if beta < 0:
                warning_msg = f"Beta value {beta:.6f} is negative. This may lead to trivial solutions."
                logger.warning(warning_msg)

            if epsilon <= 0:
                warning_msg = f"Epsilon value {epsilon:.6f} should be positive for strict inequality. Using default 1."
                logger.warning(warning_msg)
                epsilon = 1

        logger.info(
            f"Solving beta-constraint model with beta={beta:.6f}, epsilon={epsilon:.6f}"
        )
        start_time = time.time()

        # add dispersion constraint with proper type annotation
        try:
            # add quadratic dispersion constraint to the model
            dispersion_constraint: QuadraticConstraint = (
                self.p_median_model.add_constraint(
                    self.p_median_model.sum(
                        self.problem_data.distance_matrix[i, j] * self.x[i] * self.x[j]
                        for i in range(self.number_of_locations)
                        for j in range(self.number_of_locations)
                    )
                    >= beta + epsilon,
                    ctname="dispersion_constraint",
                )
            )
            logger.debug(
                f"Added dispersion constraint: sum(d_ij * x_i * x_j) >= {beta + epsilon:.6f}"
            )

            # only set time limit if current_time is provided and valid
            if current_time > 0 and current_time < self.time_limit:
                remaining_time = max(1, self.time_limit - current_time)
                try:
                    self.p_median_model.parameters.timelimit = int(remaining_time)
                    logger.debug(f"Set time limit to {int(remaining_time)} seconds")
                except Exception as e:
                    logger.warning(
                        f"Could not set time limit: {e}. Using default CPLEX time limit."
                    )
            elif current_time >= self.time_limit:
                logger.warning(
                    f"Time limit ({self.time_limit}s) already exceeded at {current_time:.2f}s"
                )
                model_solve_time = time.time() - start_time
                return (
                    None,
                    None,
                    model_solve_time,
                    None,
                    "infeasible",
                )

            else:
                logger.debug(
                    "Using default CPLEX time limit (no current_time provided)"
                )

        except Exception as e:
            error_msg = f"Failed to add dispersion constraint: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        try:
            # solve the optimization model
            logger.debug("Starting optimization solver...")
            solution: Optional[Any] = self.p_median_model.solve()
            solve_time: float = time.time() - start_time

            # check solver status and log details
            solver_status = self.p_median_model.solve_details.status
            logger.info(
                f"Solver finished with status: {solver_status}, runtime: {solve_time:.3f}s"
            )

            if solution is not None:
                # extract solution information with proper type hints
                objective_value: float = solution.get_objective_value()
                model_solve_time: float = self.p_median_model.solve_details.time

                # log solution quality metrics
                if hasattr(self.p_median_model.solve_details, "best_bound"):
                    best_bound: float = self.p_median_model.solve_details.best_bound
                    if best_bound is not None:
                        gap: float = abs(best_bound - objective_value)
                        relative_gap: float = (
                            gap / max(abs(objective_value), 1e-10) * 100
                        )
                        logger.info(
                            f"Solution quality: objective={objective_value:.6f}, "
                            f"best_bound={best_bound:.6f}, gap={relative_gap:.4f}%"
                        )

                # extract facility selection variables as numpy array
                facility_selection: np.ndarray = np.array(
                    solution.get_value_list(self.x)
                )
                if self.validate:
                    # validate solution feasibility
                    selected_facilities = np.sum(facility_selection)
                    if abs(selected_facilities - self.p_median) > 1e-6:
                        warning_msg = f"Solution may be invalid: selected {selected_facilities:.1f} facilities instead of {self.p_median}"
                        logger.warning(warning_msg)

                # calculate achieved dispersion value (sum of squared distances)
                achieved_dispersion: float = sum(
                    self.problem_data.distance_matrix[i, j]
                    * facility_selection[i]
                    * facility_selection[j]
                    for i in range(self.number_of_locations)
                    for j in range(self.number_of_locations)
                )

                # verify dispersion constraint satisfaction
                if achieved_dispersion < beta - 1e-6:
                    warning_msg = f"Achieved dispersion ({achieved_dispersion:.6f}) is below required beta ({beta:.6f})"
                    logger.warning(warning_msg)
                else:
                    logger.info(
                        f"Successfully achieved dispersion: {achieved_dispersion:.6f} >= {beta:.6f}"
                    )

                # log selected facilities for debugging
                selected_indices = np.where(facility_selection > 0.5)[0]
                logger.debug(f"Selected facilities: {selected_indices.tolist()}")
                model_solve_time = time.time() - start_time

                return (
                    facility_selection,
                    objective_value,
                    model_solve_time,
                    achieved_dispersion,
                    "feasible",
                )

            else:
                # no feasible solution found
                logger.warning(f"No feasible solution found for beta={beta:.6f}")

                # provide diagnostic information
                if hasattr(self.p_median_model.solve_details, "status_code"):
                    status_code = self.p_median_model.solve_details.status_code
                    logger.info(f"Solver status code: {status_code}")

                model_solve_time = time.time() - start_time
                return (
                    None,
                    None,
                    model_solve_time,
                    None,
                    "infeasible",
                )

        except Exception as e:
            error_msg = f"Error during optimization: {str(e)}"
            logger.error(error_msg, exc_info=True)

            model_solve_time = time.time() - start_time
            return (
                None,
                None,
                model_solve_time,
                None,
                "infeasible",
            )

        finally:
            # always remove the temporary dispersion constraint
            try:
                self.p_median_model.remove_constraint(dispersion_constraint)
                logger.debug("Successfully removed dispersion constraint")

            except Exception as e:
                warning_msg = (
                    f"Warning: Failed to remove dispersion constraint: {str(e)}"
                )
                logger.warning(warning_msg)

    def cutting_plane_model(
        self,
        beta: float,
        epsilon: float = 1,
        max_iterations: int = 10**9,
        current_time: float = 0.0,
    ) -> Tuple[np.ndarray, float, float, float, str]:
        """
        Solve the p-median problem with beta-constraint for minimum dispersion.

        Args:
            beta: Minimum required dispersion value (sum of squared distances)
            epsilon: Small tolerance value to ensure strict inequality

        Returns:
            - Tuple[np.ndarray, float, float, float]: (solution_vector, objective_value, runtime, achieved_beta)
            - str: "infeasible" if no solution exists
        """
        # validate input parameters
        if self.validate:
            if beta < 0:
                warning_msg = f"Beta value {beta:.6f} is negative. This may lead to trivial solutions."
                logger.warning(warning_msg)

            if epsilon <= 0:
                warning_msg = f"Epsilon value {epsilon:.6f} should be positive for strict inequality. Using default 1."
                logger.warning(warning_msg)
                epsilon = 1

        logger.info(
            f"Solving beta-constraint model with beta={beta:.6f}, epsilon={epsilon:.6f}"
        )
        start_time = time.time()

        try:
            # start counting wall time
            start_time = time.time()
            # add dispersion constraint with proper type annotation
            # we start with cut-plane model to approximate the quadratic term
            # solve initial problem to get starting point
            self.p_median_model.solve()
            solution = self.p_median_model.solution
            xr = np.array(solution.get_value_list(self.x))
            lowerbound = self.p_median_model.objective_value
            updated_beta = 0
            iteration = 0

            # solve the optimization model
            logger.debug("Starting cutting plane solver...")
            solution: Optional[Any] = self.p_median_model.solve()
            solve_time: float = time.time() - start_time
            # cutting planes for dispersion
            dispersion_constraints: Dict[int, LinearConstraint] = {}

            while (
                updated_beta < beta + epsilon
                and iteration < max_iterations
                and solve_time + current_time < self.time_limit
            ):
                updated_beta = self.p_median_model.objective_value

                # get gradient at current solution
                gradient: np.ndarray = self.calculate_quadratic_gradient(
                    xr, self.problem_data.distance_matrix
                )

                # add cutting plane for current solution
                dispersion_constraints[iteration] = self.p_median_model.add_constraint(
                    self.p_median_model.sum(
                        2 * gradient[i] * self.x[i]
                        for i in range(self.number_of_locations)
                    )
                    - self.calculate_quadratic_objective(
                        xr, self.problem_data.distance_matrix
                    )
                    >= beta + epsilon,
                    ctname=f"cutting_plane_{iteration}",
                )
                iteration += 1

                # resolve the model
                self.p_median_model.solve()
                solution = self.p_median_model.solution

                if solution is None:

                    model_solve_time = time.time() - start_time
                    logger.debug("Infeasibility of cutting is achieved.")
                    return (
                        None,
                        None,
                        model_solve_time,
                        None,
                        "infeasible",
                    )

                else:
                    # get new candidate solution
                    xr = np.array(solution.get_value_list(self.x))

                    # update lower bound using optimized calculation
                    updated_beta = self.calculate_quadratic_objective(
                        xr, self.problem_data.distance_matrix
                    )
                    lowerbound = self.p_median_model.objective_value
                    solve_time = time.time() - start_time

            # cutting plane algorithm completed successfully
            logger.info(
                f"Cutting plane algorithm converged after {iteration} iterations"
            )
            logger.info(
                f"Final beta value: {updated_beta:.6f}, target: {beta + epsilon:.6f}"
            )

            # extract final solution information
            objective_value: float = lowerbound
            model_solve_time: float = solve_time

            # calculate final achieved dispersion using optimized method
            achieved_dispersion: float = self.calculate_quadratic_objective(
                xr, self.problem_data.distance_matrix
            )

            # validate solution feasibility
            if self.validate:
                selected_facilities = np.sum(xr)
                if abs(selected_facilities - self.p_median) > 1e-6:
                    warning_msg = f"Solution may be invalid: selected {selected_facilities:.1f} facilities instead of {self.p_median}"
                    logger.warning(warning_msg)

                # verify dispersion constraint satisfaction
                if achieved_dispersion < beta - 1e-6:
                    warning_msg = f"Achieved dispersion ({achieved_dispersion:.6f}) is below required beta ({beta:.6f})"
                    logger.warning(warning_msg)
                else:
                    logger.info(
                        f"Successfully achieved dispersion: {achieved_dispersion:.6f} >= {beta:.6f}"
                    )

            # log selected facilities for debugging
            selected_indices = np.where(xr > 0.5)[0]
            logger.debug(f"Selected facilities: {selected_indices.tolist()}")

            return (
                xr,
                objective_value,
                model_solve_time,
                achieved_dispersion,
                "feasible",
            )

        except Exception as e:
            error_msg = f"Error during cutting plane optimization: {str(e)}"
            logger.error(error_msg, exc_info=True)
            model_solve_time = time.time() - start_time
            return (
                None,
                None,
                model_solve_time,
                None,
                "infeasible",
            )

        finally:
            # always remove all temporary cutting plane constraints in batch
            try:
                if dispersion_constraints:
                    # collect all constraint objects and remove in one batch operation
                    constraints_to_remove = list(dispersion_constraints.values())
                    self.p_median_model.remove_constraints(constraints_to_remove)
                    logger.debug(
                        f"Successfully removed {len(constraints_to_remove)} cutting plane constraints"
                    )
            except Exception as e:
                warning_msg = (
                    f"Warning: Failed to remove cutting plane constraints: {str(e)}"
                )
                logger.warning(warning_msg)
