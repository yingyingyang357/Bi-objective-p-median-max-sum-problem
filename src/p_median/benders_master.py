import time
import logging
import numpy as np
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from docplex.mp.constr import LinearConstraint, QuadraticConstraint
from typing import List, Optional, ClassVar, Dict, Union, Any
from src.data_class import ProblemData
from src.p_median.beta_model import BetaPmeanModel

# configure logger for the beta model module
logger = logging.getLogger(__name__)


class BendersMaster(BetaPmeanModel):
    """
    Master problem for p-median optimization with dispersion constraints via Benders decomposition.

    This class extends BetaPmeanModel to include Benders decomposition specific functionality
    for solving bi-objective p-median problems with dispersion constraints. It adds theta
    variables and optimality cuts for the Benders master problem formulation.

    Extends BetaPmeanModel with:
    - Theta variables for Benders decomposition
    - Lambda and pi value computation for dual solutions
    - Optimality and feasibility cut generation
    - Subproblem coordination methods
    """

    name: ClassVar[str] = "P-Median Benders Master Model"

    def __init__(
        self,
        p_median: int,
        min_distance: float,
        problem_data: ProblemData,
        validate: bool = False,
        **data,
    ) -> None:
        """
        Initialize the Benders Master Model by extending BetaPmeanModel.

        Args:
            p_median: Number of facilities to select as medians
            min_distance: Minimum required distance between selected facilities
            problem_data: Problem instance containing facilities and distance matrix
            validate: Whether to perform input validation and detailed logging
            **data: Additional configuration parameters
        """
        # initialize the parent BetaPmeanModel
        super().__init__(
            p_median=p_median,
            min_distance=min_distance,
            problem_data=problem_data,
            validate=validate,
            **data,
        )

        # optimization model instance
        self.master_model: Model = Model(name="Benders Master Model")

        # get decision variables
        # decision variables
        self.xi: List[int] = [i for i in range(self.number_of_locations)]
        self.thetak: List[int] = [k for k in range(self.number_of_locations)]

        self.x: List[Var] = self.master_model.binary_var_list(keys=self.xi, name="x")
        self.theta: List[Var] = self.master_model.continuous_var_list(
            keys=self.thetak, name="theta", lb=0
        )
        # build the master model
        # add initial constraints of selecting p facilities
        self.master_model.add_constraint(
            sum(self.x) == self.p_median, "select_p_facilities"
        )
        # add max-distance constraints between selected facilities
        if self.conflict_pairs:
            self.master_model.add_constraints(
                self.x[i] + self.x[j] <= 1 for i, j in self.conflict_pairs
            )

        # set objective to minimize total assignment cost
        self.master_model.minimize(
            self.master_model.sum(
                self.theta[k] for k in range(self.number_of_locations)
            )
        )
        if validate:
            logger.info(
                f"Extended BetaPmeanModel to Benders Master: "
                f"added {len(self.theta)} theta variables"
            )

    def get_lambda(
        self, facility_selection: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Compute lambda values for Benders decomposition dual variables.

        For each demand location k, computes lambda[k] = min{d[i,k] : x[i] = 1},
        which represents the minimum distance from any selected facility to location k.

        Mathematical Context:
            In the p-median Benders decomposition, lambda[k] represents the dual variable
            corresponding to the demand constraint at location k. It equals the shortest
            distance from location k to any open facility.

        Args:
            facility_selection: Binary solution vector where x[i] = 1 if facility i is selected.
                            Can be list or numpy array with values in [0,1].

        Returns:
            np.ndarray: Lambda values of shape (n_locations,) where lambda[k] is the minimum
                    distance from any selected facility to demand location k.

        """
        # convert to numpy array for efficient operations
        x_array: np.ndarray = np.asarray(facility_selection, dtype=float)

        # get indices of selected facilities
        selected_facilities: np.ndarray = np.where(x_array > 0.5)[0]

        # vectorized computation: extract distances from selected facilities to all locations
        # shape: (n_selected_facilities, n_locations)
        selected_distances: np.ndarray = self.problem_data.distance_matrix[
            selected_facilities, :
        ]

        # compute minimum distance from any selected facility to each location
        # axis=0 gives min over facilities for each location
        lambda_values: np.ndarray = np.min(selected_distances, axis=0)

        return lambda_values

    def get_pi(self, facility_selection: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute pi values (reduced costs) for Benders decomposition.

        For each facility-location pair (i,k), computes:
        - pi[i,k] = min(0, d[i,k] - lambda[k]) if facility i is NOT selected
        - pi[i,k] = 0 if facility i is selected

        Mathematical Context:
            Pi values represent the reduced costs in the Benders subproblem dual.
            They indicate the potential cost reduction if an inactive facility were opened.
            Negative pi values suggest beneficial facility locations for the next iteration.

        Args:
            facility_selection: Binary solution vector where x[i] = 1 if facility i is selected.

        Returns:
            np.ndarray: Pi matrix of shape (n_facilities, n_locations) where pi[i,k]
                    represents the reduced cost for assigning location k to facility i.
        """
        # get lambda values using optimized method
        lambda_values: np.ndarray = self.get_lambda(facility_selection)

        # convert facility selection to numpy array and create inactive mask
        x_array: np.ndarray = np.asarray(facility_selection, dtype=float)
        inactive_facilities_mask: np.ndarray = x_array <= 0.5

        # distance_matrix (n×n) - lambda_values (n,) → (n×n) matrix
        reduced_costs: np.ndarray = np.minimum(
            0, self.problem_data.distance_matrix - lambda_values
        )

        # zero out rows corresponding to active (selected) facilities
        # only inactive facilities can have non-zero pi values
        reduced_costs[~inactive_facilities_mask, :] = 0.0

        return reduced_costs

    def compute_upper_bound(
        self, lambda_values: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute upper bound for the p-median objective function.

        The upper bound equals sum(lambda[k]) over all demand locations k, representing
        the total assignment cost when each location is served by its nearest selected facility.

        Args:
            lambda_values: Array of lambda values from get_lambda() method.
                        Each lambda[k] is the minimum distance to location k.

        Returns:
            float: upper bound on assignment cost.
        """
        # convert to numpy array if needed and compute sum efficiently
        lambda_array: np.ndarray = np.asarray(lambda_values, dtype=float)

        # compute total assignment cost (upper bound)
        total_assignment_cost: float = float(np.sum(lambda_array))

        return total_assignment_cost

    def subproblem_model(self, facility_selection: np.ndarray) -> tuple:
        """
        Solve the p-median subproblem for given facility selection.

        This method solves the assignment subproblem to compute dual variables
        (lambda and pi values) for the Benders decomposition.

        Args:
            facility_selection: Binary array indicating which facilities are selected

        Returns:
            tuple: (lambda_values, pi_values, subproblem_objective, assignment_matrix)
        """
        # setup subproblem model
        sub = Model(name="p-median subproblem")

        # set continuous assignment variables
        y = sub.continuous_var_dict(keys=self.yik, name="y", lb=0, ub=1)

        # add demand satisfaction constraints (each location must be assigned)
        demand_constraints = dict()
        for k in range(self.number_of_locations):
            constraint = sub.add_constraint(
                sub.sum(y[i, k] for i in range(self.number_of_locations)) == 1,
                ctname=f"demand_{k}",
            )
            demand_constraints[k] = constraint

        # add facility capacity constraints (can only assign to open facilities)
        capacity_constraints = dict()
        for i in range(self.number_of_locations):
            for k in range(self.number_of_locations):
                constraint = sub.add_constraint(
                    y[i, k] <= facility_selection[i], ctname=f"capacity_{i}_{k}"
                )
                capacity_constraints[(i, k)] = constraint

        # set objective: minimize total assignment cost
        sub.minimize(
            sub.sum(
                self.problem_data.distance_matrix[i, k] * y[i, k] for i, k in self.yik
            )
        )

        # solve the subproblem
        sub.solve()

        if sub.solution is None:
            logger.error("Subproblem is infeasible")
            return None, None, float("inf"), None

        # extract solution information
        sub_solution = sub.solution
        sub_objective = sub.objective_value

        # extract dual variables (lambda values from demand constraints)
        lambda_values = np.array(
            [
                (
                    demand_constraints[k].dual_value
                    if demand_constraints[k].dual_value is not None
                    else 0.0
                )
                for k in range(self.number_of_locations)
            ]
        )

        # extract dual variables (pi values from capacity constraints)
        pi_values = np.zeros((self.number_of_locations, self.number_of_locations))
        for i, k in self.yik:
            dual_val = capacity_constraints[(i, k)].dual_value
            pi_values[i, k] = dual_val if dual_val is not None else 0.0

        # extract primal assignment solution
        assignment_matrix = np.zeros(
            (self.number_of_locations, self.number_of_locations)
        )
        for i, k in self.yik:
            assignment_matrix[i, k] = sub_solution.get_value(y[i, k])

        return lambda_values, pi_values, sub_objective, assignment_matrix

    def _validate_beta_epsilon_inputs(self, beta: float, epsilon: float) -> float:
        """
        Validate and adjust beta and epsilon input parameters.

        Args:
            beta: Minimum required dispersion value
            epsilon: Small tolerance value for strict inequality

        Returns:
            float: Validated (and potentially adjusted) epsilon value

        Raises:
            ValueError: If parameters are invalid and validation is enabled
        """
        if not self.validate:
            return epsilon

        if beta < 0:
            warning_msg = f"Beta value {beta:.6f} is negative. This may lead to trivial solutions."
            logger.warning(warning_msg)

        if epsilon <= 0:
            warning_msg = (
                f"Epsilon value {epsilon:.6f} should be positive for strict inequality. "
                f"Using default 0.01."
            )
            logger.warning(warning_msg)
            epsilon = 1

        if epsilon >= beta * 0.1 and beta > 0:
            warning_msg = (
                f"Epsilon ({epsilon:.6f}) is large relative to beta ({beta:.6f}). "
                f"Consider using smaller epsilon."
            )
            logger.warning(warning_msg)

        return epsilon

    def _validate_solution_output(
        self, facility_selection: np.ndarray, achieved_dispersion: float, beta: float
    ) -> None:
        """
        Validate solution feasibility and constraint satisfaction.

        Args:
            facility_selection: Binary array indicating selected facilities
            achieved_dispersion: Actual dispersion value achieved
            beta: Required minimum dispersion value
        """
        if not self.validate:
            return

        # validate solution feasibility - correct number of facilities selected
        selected_facilities_count = np.sum(facility_selection > 0.5)
        if abs(selected_facilities_count - self.p_median) > 1e-6:
            warning_msg = (
                f"Solution may be invalid: selected {selected_facilities_count:.1f} "
                f"facilities instead of {self.p_median}"
            )
            logger.warning(warning_msg)

        # verify dispersion constraint satisfaction
        if achieved_dispersion < beta - 1e-6:
            warning_msg = (
                f"Achieved dispersion ({achieved_dispersion:.6f}) is below "
                f"required beta ({beta:.6f})"
            )
            logger.warning(warning_msg)

        else:
            logger.info(
                f"Successfully achieved dispersion: {achieved_dispersion:.6f} >= {beta:.6f}"
            )

        # log selected facilities for debugging
        selected_indices = np.where(facility_selection > 0.5)[0]
        logger.debug(f"Selected facilities: {selected_indices.tolist()}")

    def quadratic_model(
        self, beta: float, epsilon: float, current_time: float = 0.0
    ) -> None:
        """
        We add beta quadratic constraints for max-diversity to the master model.
        Σ d_ij * x_i * x_j ≥ β + ε (dispersion constraint)
        """
        # validate input parameters using centralized validation
        epsilon = self._validate_beta_epsilon_inputs(beta, epsilon)

        logger.info(
            f"Solving beta-constraint model with beta={beta:.6f}, epsilon={epsilon:.6f}"
        )
        # set time limit
        if current_time < self.time_limit:
            # register start time
            start_time = time.time()

            # add quadratic dispersion constraint to the model
            dispersion_constraint: QuadraticConstraint = (
                self.master_model.add_constraint(
                    self.master_model.sum(
                        self.problem_data.distance_matrix[i, j] * self.x[i] * self.x[j]
                        for i in range(self.number_of_locations)
                        for j in range(self.number_of_locations)
                    )
                    >= beta + epsilon,
                    ctname="dispersion_constraint" + str((beta, epsilon)),
                )
            )
            logger.debug(
                f"Added dispersion constraint: sum(d_ij * x_i * x_j) >= {beta + epsilon:.6f}"
            )
            self.master_model.set_time_limit(int(self.time_limit - current_time))

            self.master_model.solve()
            solution = self.master_model.solution
            if solution is None:
                logger.debug(
                    "Infeasibility of quadratic dispersion constraint is achieved."
                )
            else:
                facility_selection = np.array(solution.get_value_list(self.x))
                achieved_dispersion = BetaPmeanModel.calculate_quadratic_objective(
                    facility_selection, self.problem_data.distance_matrix
                )
                logger.info(
                    f"Achieved dispersion after adding quadratic constraint: {achieved_dispersion:.6f}"
                )
                # validate solution output
                self._validate_solution_output(
                    facility_selection, achieved_dispersion, beta
                )

            # always remove the dispersion constraint after solving
            self.master_model.remove_constraint(dispersion_constraint)
            return (
                facility_selection,
                self.master_model.objective_value,
                time.time() - start_time,
                achieved_dispersion,
                "feasible",
            )
        else:
            logger.warning(
                f"Time limit exceeded: current_time={current_time:.2f}, time_limit={self.time_limit}"
            )
            return (
                None,
                None,
                0.0,
                None,
                "infeasible",
            )

    def remove_dispersion_constraint(self) -> None:
        """
        Remove any dispersion constraint from the master model.
        """
        dispersion_constraints = [
            constraint
            for constraint in self.master_model.iter_constraints()
            if constraint.name is not None
            and constraint.name.startswith("dispersion_constraint")
        ]

        self.master_model.remove_constraints(dispersion_constraints)
        logger.info("Successfully removed dispersion constraints")

    def remove_subproblem_constraint(self) -> None:
        """
        Remove any dispersion constraint from the master model.
        """
        subproblem_constraints = [
            constraint
            for constraint in self.master_model.iter_constraints()
            if constraint.name is not None
            and constraint.name.startswith("subproblem_constraint")
        ]

        self.master_model.remove_constraints(subproblem_constraints)
        logger.info("Successfully removed subproblem constraints")

    def cutting_plane_model(
        self,
        beta: float,
        epsilon: float,
        max_iterations: int = 10**9,
        current_time: float = 0.0,
    ) -> None:
        """
        Solve the p-median problem with beta-constraint for minimum dispersion.
        Use cutting plane method to iteratively add linear approximations of the quadratic

        Args:
            beta: Minimum required dispersion value (sum of squared distances)
            epsilon: Small tolerance value to ensure strict inequality

        Returns:
            - Tuple[np.ndarray, float, float, float]: (solution_vector, objective_value, runtime, achieved_beta)
            - str: "infeasible" if no solution exists
        """
        # validate input parameters using centralized validation
        epsilon = self._validate_beta_epsilon_inputs(beta, epsilon)

        logger.info(
            f"Solving beta-constraint model with beta={beta:.6f}, epsilon={epsilon:.6f}"
        )

        try:
            start_time = time.time()
            solve_time = 0.0
            # add dispersion constraint with proper type annotation
            # we start with cut-plane model to approximate the quadratic term
            # solve initial problem to get starting point

            self.master_model.solve()
            solution = self.master_model.solution
            xr = np.array(solution.get_value_list(self.x))

            lowerbound = self.master_model.objective_value
            updated_beta = 0
            iteration = 0

            # solve the optimization model
            logger.debug("Starting cutting plane solver...")
            solution: Optional[Any] = self.master_model.solve()

            # cutting planes for dispersion
            dispersion_constraints: Dict[int, LinearConstraint] = {}

            while (
                updated_beta < beta + epsilon
                and iteration < max_iterations
                and solve_time + current_time < self.time_limit
            ):
                updated_beta = self.master_model.objective_value

                # get gradient at current solution
                gradient: np.ndarray = BetaPmeanModel.calculate_quadratic_gradient(
                    xr, self.problem_data.distance_matrix
                )

                # add cutting plane for current solution
                dispersion_constraints[iteration] = self.master_model.add_constraint(
                    self.master_model.sum(
                        2 * gradient[i] * self.x[i]
                        for i in range(self.number_of_locations)
                    )
                    - BetaPmeanModel.calculate_quadratic_objective(
                        xr, self.problem_data.distance_matrix
                    )
                    >= beta + epsilon,
                    ctname=f"cutting_plane_{iteration}",
                )
                iteration += 1

                # resolve the model
                self.master_model.solve()
                solution = self.master_model.solution

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
                    updated_beta = BetaPmeanModel.calculate_quadratic_objective(
                        xr, self.problem_data.distance_matrix
                    )
                    lowerbound = self.master_model.objective_value
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
            achieved_dispersion: float = BetaPmeanModel.calculate_quadratic_objective(
                xr, self.problem_data.distance_matrix
            )

            # validate solution output using centralized validation
            self._validate_solution_output(xr, achieved_dispersion, beta)

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
                    self.master_model.remove_constraints(constraints_to_remove)
                    logger.debug(
                        f"Successfully removed {len(dispersion_constraints)} cutting plane constraints"
                    )
            except Exception as e:
                warning_msg = (
                    f"Warning: Failed to remove cutting plane constraints: {str(e)}"
                )
                logger.warning(warning_msg)
