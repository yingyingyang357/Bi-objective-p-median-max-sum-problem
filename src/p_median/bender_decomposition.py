import time
import logging
import numpy as np
import cplex
from cplex.callbacks import LazyConstraintCallback
from typing import Tuple, ClassVar, Union
from src.data_class import ProblemData
from src.p_median.benders_master import BendersMaster


# configure logger for the BendersDecompositionModel module
logger = logging.getLogger(__name__)


# Create the callback class with correct initialization
class BendersCallback(LazyConstraintCallback):
    def __init__(self, env):
        super().__init__(env)
        self.master_problem = None
        self.beta = None
        self.epsilon = None
        self.cutting_plane = None
        self.x_indices = None
        self.theta_indices = None
        self.n_locations = 0
        self.cut_count = 0
        self.callback_calls = 0

    def set_data(
        self,
        master_problem,
        beta,
        epsilon,
        cutting_plane,
        x_indices,
        theta_indices,
    ):
        self.master_problem = master_problem
        self.beta = beta
        self.epsilon = epsilon
        self.cutting_plane = cutting_plane
        self.x_indices = x_indices
        self.theta_indices = theta_indices
        self.n_locations = len(x_indices)

    def __call__(self):
        self.callback_calls += 1

        try:
            # Extract current solution values
            x_vals = np.array(self.get_values(self.x_indices))

            # Check if solution is integer
            if not np.all(np.abs(x_vals - np.round(x_vals)) < 1e-6):
                return

            # Validate p-median solution
            n_selected = int(np.sum(x_vals > 0.5))
            if n_selected != self.master_problem.p_median:
                return

            # Get Benders subproblem information
            lambda_vals, pi_matrix, _, _ = self.master_problem.subproblem_model(x_vals)

            # Add Benders optimality cuts
            for k in range(self.n_locations):
                coefficients = [1.0]  # theta[k] coefficient
                variable_indices = [self.theta_indices[k]]

                # Add x[i] coefficients (negative of pi values)
                for i in range(self.n_locations):
                    coefficients.append(-pi_matrix[i, k])
                    variable_indices.append(self.x_indices[i])

                self.add(
                    constraint=cplex.SparsePair(ind=variable_indices, val=coefficients),
                    sense="G",
                    rhs=lambda_vals[k],
                )
                self.cut_count += 1

            # Add quadratic tangent cut if requested
            if self.cutting_plane:
                gradient = 2 * self.master_problem.calculate_quadratic_gradient(
                    x_vals, self.master_problem.problem_data.distance_matrix
                )
                dispersion_value = self.master_problem.calculate_quadratic_objective(
                    x_vals,
                    self.master_problem.problem_data.distance_matrix,
                )

                tangent_rhs = self.beta + self.epsilon + dispersion_value

                self.add(
                    constraint=cplex.SparsePair(
                        ind=self.x_indices, val=gradient.tolist()
                    ),
                    sense="G",
                    rhs=tangent_rhs,
                )
                self.cut_count += 1

        except Exception as e:
            logger.error(f"Callback error: {e}")


class BendersDecompositionModel(BendersMaster):
    """
    Benders Decomposition Model for p-median optimization with dispersion constraints.
    """

    name: ClassVar[str] = "P-Median Benders Decomposition Model"
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
        Initialize the Benders Decomposition Model for bi-objective p-median optimization.

        This class extends BendersMaster to provide a complete Benders decomposition
        implementation with both master problem and subproblem coordination.

        Args:
            p_median: Number of facilities to select as medians
            min_distance: Minimum required distance between selected facilities
            problem_data: Problem instance containing facilities and distance matrix
            validate: Whether to perform input validation and detailed logging
            **data: Additional configuration parameters including:
                - time_limit (int): Maximum computation time in seconds
                - max_iterations (int): Maximum Benders iterations
                - tolerance (float): Convergence tolerance for dual bounds

        Note:
            This class inherits all methods from BendersMaster, including:
            - get_lambda(): Compute dual variables
            - get_pi(): Compute reduced costs
            - quadratic_model(): Direct quadratic approach
            - cutting_plane_model(): Linear approximation approach
            - Constraint management methods
        """
        # call parent constructor to initialize the master problem
        super().__init__(
            p_median=p_median,
            min_distance=min_distance,
            problem_data=problem_data,
            validate=validate,
            **data,
        )

        # update name for the decomposition model
        self.name = "p-Median Benders Decomposition Model"

        # benders-specific parameters
        self.max_benders_iterations = data.get("max_iterations", 10**9)
        self.convergence_tolerance = data.get("tolerance", 1e-6)

        if validate:
            logger.info(
                f"Initialized Benders Decomposition Model: "
                f"max_iterations={self.max_benders_iterations}, "
                f"tolerance={self.convergence_tolerance:.2e}"
            )

    def balinski_cut(
        self,
        beta: float,
        epsilon: float,
        current_time: float = 0.0,
        cutting_plane: bool = True,
        cutting_plane_iter: int = 10**9,
    ) -> None:
        """
        Placeholder for Balinski cut implementation in Benders decomposition.

        This method is intended to add Balinski cuts to the master problem
        to improve convergence. The actual implementation will depend on
        the specific structure of the p-median problem and its dual formulation.
        """
        iteration_count = 0
        lower_bound = float("-inf")
        upper_bound = float("inf")
        runtime = current_time

        while (
            runtime < self.time_limit
            and iteration_count < self.max_benders_iterations
            and upper_bound - lower_bound >= self.convergence_tolerance
        ):
            start_time = time.time()

            # solve the master problem with dispersion constraints
            if cutting_plane:
                xr, objective_value, model_solve_time, achieved_dispersion, _ = (
                    self.cutting_plane_model(
                        beta=beta,
                        epsilon=epsilon,
                        max_iterations=cutting_plane_iter,
                        current_time=runtime,
                    )
                )

            else:
                xr, objective_value, model_solve_time, achieved_dispersion, _ = (
                    self.quadratic_model(
                        beta=beta,
                        epsilon=epsilon,
                        current_time=runtime,
                    )
                )

            # balinski method to get lambda and pi and upperbound
            lambda_l = self.get_lambda(facility_selection=xr)
            pi_l = self.get_pi(facility_selection=xr)

            # update the bounds
            upper_bound = min(
                self.compute_upper_bound(lambda_values=lambda_l), upper_bound
            )
            lower_bound = max(objective_value, lower_bound)

            # add all optimality cuts to the master model in batch
            self.master_model.add_constraints(
                (
                    self.theta[k]
                    >= lambda_l[k]
                    + self.master_model.sum(
                        pi_l[i, k] * self.x[i] for i in range(self.number_of_locations)
                    ),
                    f"subproblem_constraint_{iteration_count}_{k}",
                )
                for k in range(self.number_of_locations)
            )
            iteration_count += 1
            runtime += time.time() - start_time

        logger.info(
            f"Balinski cut method completed after {iteration_count} iterations. "
            f"Final bounds: LB={lower_bound:.6f}, UB={upper_bound:.6f}"
        )

        # remove all subproblem constraints after completion
        self.remove_subproblem_constraint()

        return (
            xr,
            objective_value,
            runtime,
            achieved_dispersion,
            "feasible" if upper_bound < float("inf") else "infeasible",
        )

    def branch_and_cut_benders(
        self,
        beta: float,
        epsilon: float,
        current_time: float = 0.0,
        cutting_plane: bool = True,
    ) -> Union[Tuple[np.ndarray, float, float, float, float], str]:
        """
        Branch-and-cut algorithm with Benders lazy constraints for p-median optimization.

        This method implements a branch-and-cut approach that uses lazy constraint callbacks
        to dynamically add Benders optimality cuts and quadratic tangent cuts during the
        solution process, providing tighter bounds and faster convergence.

        Args:
            beta: Minimum required dispersion value for beta-constraint
            epsilon: Small tolerance value for strict inequality constraint
            current_time: Current elapsed time (used for time limit management)
            cutting_plane: Whether to use cutting plane approach

        Returns:
            Union of:
            - Tuple[np.ndarray, float, float, float, float]:
            (solution_vector, lower_bound, upper_bound, runtime, achieved_dispersion)
            - str: "infeasible" if no solution exists or solver error occurs

        Mathematical Context:
            The branch-and-cut combines:
            1. Branch-and-bound tree search for integer variables
            2. Lazy constraint generation for Benders optimality cuts
            3. Quadratic tangent cuts for dispersion constraint linearization
        """
        # validate inputs using centralized validation
        epsilon = self._validate_beta_epsilon_inputs(beta, epsilon)

        start_time = time.time()
        remaining_time = self.time_limit - current_time

        try:
            # Create a new CPLEX model instance for branch-and-cut
            branch_cut_model = cplex.Cplex()
            branch_cut_model.set_problem_type(cplex.Cplex.problem_type.MILP)

            # Set solver parameters for branch-and-cut performance
            branch_cut_model.parameters.timelimit.set(int(remaining_time))
            branch_cut_model.parameters.preprocessing.presolve.set(
                0
            )  # disable presolve
            branch_cut_model.parameters.mip.strategy.search.set(
                1
            )  # traditional branch-and-cut

            logger.info(
                f"Starting branch-and-cut with beta={beta:.6f}, epsilon={epsilon:.6f}"
            )

            # Add binary facility selection variables: x[0..n-1]
            x_names = [f"x_{i}" for i in range(self.number_of_locations)]
            branch_cut_model.variables.add(
                names=x_names,
                types=["B"] * self.number_of_locations,  # Binary variables
            )

            # Add continuous assignment cost variables: theta[0..n-1]
            theta_names = [f"theta_{i}" for i in range(self.number_of_locations)]
            branch_cut_model.variables.add(
                names=theta_names,
                types=["C"] * self.number_of_locations,  # Continuous variables
                lb=[0.0] * self.number_of_locations,  # Non-negative bounds
            )

            # Get variable indices for callback interface
            x_indices = list(range(self.number_of_locations))
            theta_indices = list(
                range(self.number_of_locations, 2 * self.number_of_locations)
            )

            # Add p-median constraint: sum(x_i) = p
            branch_cut_model.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=x_indices, val=[1.0] * self.number_of_locations
                    )
                ],
                senses=["E"],  # Equality constraint
                rhs=[self.p_median],
                names=["p_median_constraint"],
            )

            # Add conflict constraints: x_i + x_j <= 1 when distance[i,j] < min_distance
            conflict_count = 0
            for i, j in self.conflict_pairs:

                branch_cut_model.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(
                            ind=[x_indices[i], x_indices[j]], val=[1.0, 1.0]
                        )
                    ],
                    senses=["L"],  # Less than or equal
                    rhs=[1.0],
                    names=[f"conflict_{i}_{j}"],
                )
                conflict_count += 1

            logger.debug(f"Added {conflict_count} conflict constraints")

            # Set objective: minimize sum(theta_i) - this will be the assignment cost
            # Coefficients: 0 for x variables, 1 for theta variables
            objective_coefficients = [0.0] * self.number_of_locations + [
                1.0
            ] * self.number_of_locations
            branch_cut_model.objective.set_sense(
                branch_cut_model.objective.sense.minimize
            )
            branch_cut_model.objective.set_linear(
                list(zip(range(2 * self.number_of_locations), objective_coefficients))
            )

            # Register the callback with CPLEX - uses the global BendersCallback class
            callback = branch_cut_model.register_callback(BendersCallback)

            # Set callback data
            callback.set_data(
                master_problem=self,
                beta=beta,
                epsilon=epsilon,
                cutting_plane=cutting_plane,
                x_indices=x_indices,
                theta_indices=theta_indices,
            )

            logger.debug("Registered Benders lazy constraint callback")

            # Solve the branch-and-cut model
            try:
                branch_cut_model.solve()

            except Exception as solve_error:
                logger.error(f"CPLEX solver error during branch-and-cut: {solve_error}")
                return "infeasible"

            # Calculate runtime
            runtime = time.time() - start_time

            # Extract solution if available
            try:
                # Check if solution is available
                solution_status = branch_cut_model.solution.get_status()

                if solution_status in [
                    branch_cut_model.solution.status.optimal,
                    branch_cut_model.solution.status.MIP_optimal,
                ]:

                    # Extract facility selection solution
                    x_solution = np.array(branch_cut_model.solution.get_values(x_names))

                    # Get objective bounds
                    lower_bound = branch_cut_model.solution.get_objective_value()

                    try:
                        upper_bound = branch_cut_model.solution.MIP.get_best_objective()
                    except Exception:
                        upper_bound = lower_bound

                    # Calculate achieved dispersion using optimized method
                    achieved_dispersion = self.calculate_quadratic_objective(
                        x_solution, self.problem_data.distance_matrix
                    )

                    # Calculate optimality gap
                    if abs(upper_bound) > 1e-6:
                        gap_percentage = (
                            100 * (upper_bound - lower_bound) / abs(upper_bound)
                        )
                    else:
                        gap_percentage = 0.0

                    logger.info(
                        f"Branch-and-cut completed successfully in {runtime:.2f}s. "
                        f"Objective: {lower_bound:.6f}, Gap: {gap_percentage:.2f}%, "
                        f"Cuts added: {callback.cut_count}"
                    )

                    # Validate solution if requested
                    if self.validate:
                        self._validate_solution_output(
                            x_solution, achieved_dispersion, beta
                        )

                    return (
                        x_solution,
                        lower_bound,
                        upper_bound,
                        runtime,
                        achieved_dispersion,
                    )

                else:
                    logger.warning(
                        f"Branch-and-cut found no solution. Status: {solution_status}"
                    )
                    return "infeasible"

            except Exception as extraction_error:
                logger.error(
                    f"Failed to extract solution from branch-and-cut: {extraction_error}"
                )
                return "infeasible"

        except Exception as model_error:
            logger.error(f"Branch-and-cut model creation failed: {model_error}")
            return "infeasible"
