import time
import logging
import numpy as np
from typing import ClassVar, Dict, Union, Literal
from src.data_class import ProblemData
from src.p_median.p_median_solver import PMedianOptimiser

# configure logger for the beta model module
logger = logging.getLogger(__name__)


class BiObjectiveEpsilonMethod:
    """
    Epsilon-constraint method for solving bi-objective p-median optimization problems.

    This class implements the epsilon-constraint method (also known as the parametric method)
    for generating the complete Pareto frontier of bi-objective p-median problems. The method
    systematically varies one objective as a constraint while optimizing the other, producing
    a set of non-dominated (Pareto-optimal) solutions.

    Bi-Objective Problem Formulation:
    - Objective 1: Minimize total assignment cost (classical p-median)
    - Objective 2: Maximize facility dispersion (sum of squared distances between facilities)

    The epsilon-constraint approach:
    1. Converts the dispersion objective into a constraint: dispersion ≥ β (beta)
    2. Optimizes the p-median objective subject to this dispersion constraint
    3. Incrementally increases β to generate different Pareto-optimal solutions
    4. Continues until no feasible solutions exist or time limit is reached

    Solution Methods:
    - Direct quadratic programming: Exact solutions using CPLEX QP solver
    - Cutting plane approximation: Linear approximation for faster solving
    - Benders decomposition: Decomposed approach (future implementation)

    Applications:
    - Facility location with diversity requirements
    - Service network design with coverage and dispersion trade-offs
    - Emergency service placement balancing accessibility and redundancy

    Attributes:
        p_median (int): Number of facilities to select
        min_distance (float): Minimum required distance between selected facilities
        problem_data (ProblemData): Instance containing facilities and distance matrix
        validate (bool): Enable input validation and detailed logging
        time_limit (int): Maximum total computation time in seconds

    Returns:
        Dict: Collection of Pareto-optimal solutions with their characteristics:
            - solution_vector: Binary array indicating selected facilities
            - objective_value: P-median cost (total assignment cost)
            - achieved_beta: Actual dispersion value achieved
            - runtime: Individual solution time

    Example:
        >>> epsilon_solver = BiObjectiveEpsilonMethod(
        ...     p_median=5, min_distance=10.0, problem_data=data
        ... )
        >>> pareto_solutions = epsilon_solver.epsilon_method(cuttingplane=True)
        >>> print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

    """

    name: ClassVar[str] = "P-Median Benders Master Model"
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
        Initialize the Bi-Objective Epsilon Method solver.

        Args:
            p_median (int): Number of facilities to select as medians/centers
            min_distance (float): Minimum required distance between any two selected facilities
            problem_data (ProblemData): Problem instance containing facilities and distance matrix
            validate (bool, optional): Enable input validation and detailed logging. Defaults to False.
            **data: Additional configuration parameters including:
                - time_limit (int): Maximum total computation time in seconds

        Raises:
            ValueError: If p_median <= 0 or exceeds number of available facilities
            ValueError: If min_distance < 0
            ValueError: If problem_data is None or invalid

        Note:
            The solver will generate Pareto-optimal solutions by systematically varying
            the dispersion constraint parameter β from 0 until infeasibility is reached.
        """
        # store all instance attributes immediately after validation
        self.problem_data = problem_data
        self.p_median = p_median
        self.min_distance = min_distance
        self.validate = validate
        self.time_limit = data.get("time_limit", self.time_limit)
        self.number_of_locations = problem_data.number_of_facilities

    def epsilon_method(
        self,
        method: Literal[
            "quadratic", "cutting_plane", "benders", "branch_cut_benders"
        ] = "quadratic",
        epsilon: float = 1,
        max_iterations: int = 10**9,
        cutting_plane: bool = True,
    ) -> Dict[int, Dict[str, Union[np.ndarray, float]]]:
        """
        Generate the complete Pareto frontier using the epsilon-constraint method with flexible solving methods.

        This method systematically generates all Pareto-optimal solutions for the bi-objective
        p-median problem by incrementally constraining the dispersion objective while optimizing
        the assignment cost objective using the selected solving method.

        Args:
            method: Solving method to use:
                - "quadratic": Direct quadratic programming (exact, slower)
                - "cutting_plane": Linear cutting plane approximation (faster, approximate)
                - "benders": Classical Benders decomposition
                - "branch_cut_benders": Branch-and-cut with Benders lazy constraints
            epsilon: Tolerance parameter for strict inequality constraints
            max_iterations: Maximum iterations per subproblem (for iterative methods)
            cutting_plane: Whether to use cutting plane constraints (for Benders methods)

        Returns:
            Dictionary of Pareto-optimal solutions with their characteristics
        """

        num_pareto_solutions = 0
        solutions_dict = {}

        # Set individual solution time limit (reserve time for multiple solutions)
        individual_time_limit = self.time_limit

        # Initialize the p-median optimizer
        optimizer = PMedianOptimiser()

        logger.info(f"Starting epsilon-constraint method with {method} solver")
        logger.info(f"Individual solution time limit: {individual_time_limit}s")

        # Start epsilon-constraint iterations
        beta_value = 0.0  # Initial beta value (no dispersion requirement)
        time_elapsed = 0.0
        iteration = 0
        start_time = time.time()

        while time_elapsed < self.time_limit and iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Solving with beta >= {beta_value:.6f}")
            start_time_iter = time.time()

            # Solve p-median problem with current beta constraint using selected method
            try:
                if method == "quadratic":
                    result = optimizer.solve_quadratic_model(
                        problem=self.problem_data,
                        p_median=self.p_median,
                        min_distance=self.min_distance,
                        beta=beta_value,
                        epsilon=epsilon,
                        current_time=time_elapsed,
                        validation=self.validate,
                    )

                elif method == "cutting_plane":
                    result = optimizer.solve_cutting_plane_model(
                        problem=self.problem_data,
                        p_median=self.p_median,
                        min_distance=self.min_distance,
                        beta=beta_value,
                        epsilon=epsilon,
                        max_iterations=max_iterations,
                        current_time=time_elapsed,
                        validation=self.validate,
                    )
                elif method == "benders":
                    result = optimizer.solve_benders_decomposition(
                        problem=self.problem_data,
                        p_median=self.p_median,
                        min_distance=self.min_distance,
                        beta=beta_value,
                        epsilon=epsilon,
                        max_iterations=max_iterations,
                        current_time=time_elapsed,
                        validation=self.validate,
                        cutting_plane=cutting_plane,
                    )

                elif method == "branch_cut_benders":
                    result = optimizer.solve_branch_and_cut_benders(
                        problem=self.problem_data,
                        p_median=self.p_median,
                        min_distance=self.min_distance,
                        beta=beta_value,
                        epsilon=epsilon,
                        current_time=time_elapsed,
                        cutting_plane=cutting_plane,
                        validation=self.validate,
                    )

                else:
                    raise ValueError(f"Unknown method: {method}")

            except Exception as e:
                logger.error(
                    f"Error in iteration {iteration} with method {method}: {e}"
                )
                break

            individual_time_limit = individual_time_limit - (time.time() - start_time)
            time_elapsed = time.time() - start_time
            iteration_time = time.time() - start_time_iter

            # Check if solution was found
            if result["status"] not in ["optimal", "feasible"]:
                logger.info(
                    f"Epsilon method terminated at iteration {iteration}: {result['status']}"
                )
                break
            else:
                # Extract solution information with detailed debugging
                logger.debug(f"Result keys: {result.keys()}")
                logger.debug(f"Full result: {result}")

                solution_vector = result.get("solution")
                objective_value = result.get("objective_value")
                achieved_beta = result.get("achieved_dispersion")

                # Check if we have a valid solution
                if solution_vector is None or objective_value is None:
                    logger.warning(
                        f"Invalid solution at iteration {iteration}: "
                        f"solution_vector={solution_vector is not None}, "
                        f"objective_value={objective_value is not None}, terminating"
                    )
                    break

                # Debug: Log the raw achieved_beta value
                logger.debug(
                    f"Raw achieved_beta value: {achieved_beta}, type: {type(achieved_beta)}"
                )

                # Check if achieved_beta is None (natural termination - Pareto frontier complete)
                if achieved_beta is None:
                    logger.info(
                        f"Pareto frontier exploration completed at iteration {iteration}: "
                        f"No higher dispersion solutions exist. Found {num_pareto_solutions} Pareto-optimal solutions."
                    )
                    break

                # Ensure achieved_beta is a valid number
                try:
                    achieved_beta_float = float(achieved_beta)
                except (TypeError, ValueError) as e:
                    logger.info(
                        f"Pareto frontier exploration completed at iteration {iteration}: "
                        f"Cannot convert dispersion value '{achieved_beta}' to float: {e}. "
                        f"Found {num_pareto_solutions} Pareto-optimal solutions."
                    )
                    break

                # Update achieved_beta to the validated float value
                achieved_beta = achieved_beta_float

                # Check for improvement in beta (dispersion) - now safe to compare
                if (
                    num_pareto_solutions > 0
                    and abs(achieved_beta - beta_value) < epsilon
                ):
                    logger.info(
                        f"Pareto frontier exploration completed at iteration {iteration}: "
                        f"No significant improvement in dispersion ({achieved_beta:.6f} vs {beta_value:.6f}). "
                        f"Found {num_pareto_solutions} Pareto-optimal solutions."
                    )
                    break

                # Record new Pareto solution               
                if num_pareto_solutions == 0:
                    # First solution
                    lb0 = objective_value
                    num_pareto_solutions += 1
                    solutions_dict[num_pareto_solutions] = {
                        "solution_vector": solution_vector,
                        "objective_value": objective_value,
                        "achieved_beta": achieved_beta,
                        "iter_runtime": iteration_time,
                        "method": method,
                        "iteration": iteration,
                        "beta_target": beta_value,
                        "selected_facilities": result.get("selected_facilities", []),
                        "num_pareto_solutions": num_pareto_solutions,
                        "total_runtime": time_elapsed,
                    }
                else:
                    # If LB improved by at least epsilon, append a new Pareto point
                    if objective_value >= (lb0 + epsilon):
                        num_pareto_solutions += 1
                        solutions_dict[num_pareto_solutions] = {
                            "solution_vector": solution_vector,
                            "objective_value": objective_value,
                            "achieved_beta": achieved_beta,
                            "iter_runtime": iteration_time,
                            "method": method,
                            "iteration": iteration,
                            "beta_target": beta_value,
                            "selected_facilities": result.get("selected_facilities", []),
                            "num_pareto_solutions": num_pareto_solutions,
                            "total_runtime": time_elapsed,
                        }
                        lb0 = objective_value
                    else:
                        # No significant improvement: refine the last recorded point (overwrite)
                        solutions_dict[num_pareto_solutions].update({
                            "solution_vector": solution_vector,
                            "objective_value": objective_value,
                            "achieved_beta": achieved_beta,
                            "iter_runtime": iteration_time,
                            "method": method,
                            "iteration": iteration,
                            "beta_target": beta_value,
                            "selected_facilities": result.get("selected_facilities", []),
                            "num_pareto_solutions": num_pareto_solutions,
                            "total_runtime": time_elapsed,
                        })

                # Update beta for next iteration (achieved dispersion becomes new lower bound)
                beta_value = achieved_beta + epsilon

                # Check time limit
                if time_elapsed >= self.time_limit:
                    logger.warning(
                        f"Time limit reached after {num_pareto_solutions} solutions"
                    )
                    break

        total_time = time.time() - start_time
        logger.info(
            f"Epsilon-constraint method completed: {num_pareto_solutions} Pareto solutions found "
            f"in {total_time:.2f}s using {method} method"
        )

        return solutions_dict

    def analyze_pareto_frontier(
        self, solutions_dict: Dict[int, Dict]
    ) -> Dict[str, any]:
        """
        Analyze the generated Pareto frontier and provide summary statistics.

        Args:
            solutions_dict: Dictionary of Pareto solutions from epsilon_method()

        Returns:
            Dictionary with analysis results including ranges, trade-offs, and statistics
        """
        if not solutions_dict:
            return {"status": "no_solutions", "message": "No Pareto solutions found"}

        # Extract objective values and dispersions
        objectives = [sol["objective_value"] for sol in solutions_dict.values()]
        dispersions = [sol["achieved_beta"] for sol in solutions_dict.values()]
        runtimes = [sol["runtime"] for sol in solutions_dict.values()]

        analysis = {
            "num_solutions": len(solutions_dict),
            "objective_range": {
                "min": min(objectives),
                "max": max(objectives),
                "span": max(objectives) - min(objectives),
            },
            "dispersion_range": {
                "min": min(dispersions),
                "max": max(dispersions),
                "span": max(dispersions) - min(dispersions),
            },
            "runtime_stats": {
                "total": sum(runtimes),
                "average": sum(runtimes) / len(runtimes),
                "min": min(runtimes),
                "max": max(runtimes),
            },
            "trade_off_ratio": None,
        }

        # Calculate trade-off ratio (change in objective per unit change in dispersion)
        if len(solutions_dict) > 1:
            obj_span = analysis["objective_range"]["span"]
            disp_span = analysis["dispersion_range"]["span"]
            if disp_span > 0:
                analysis["trade_off_ratio"] = obj_span / disp_span

        return analysis

    def export_pareto_solutions(
        self, solutions_dict: Dict[int, Dict], filename: str = None
    ) -> str:
        """
        Export Pareto solutions to CSV file for analysis and visualization.

        Args:
            solutions_dict: Dictionary of Pareto solutions
            filename: Optional filename (will generate default if not provided)

        Returns:
            String path to the exported file
        """
        import pandas as pd
        from datetime import datetime

        if not solutions_dict:
            raise ValueError("No solutions to export")

        # Prepare data for DataFrame
        export_data = []
        for sol_id, solution in solutions_dict.items():
            selected_facilities = solution.get("selected_facilities", [])
            export_data.append(
                {
                    "Method": solution.get("method", "unknown"),
                    "Solution_ID": sol_id,
                    "Objective_Value": solution["objective_value"],
                    "Achieved_Beta": solution["achieved_beta"],
                    "Iter_Runtime": solution["iter_runtime"],
                    "Iteration": solution.get("iteration", sol_id),
                    "Beta_Target": solution.get("beta_target", 0),
                    "Num_Selected_Facilities": len(selected_facilities),
                    "Selected_Facilities": str(selected_facilities),
                    "Num_Pareto_Solutions": solution["num_pareto_solutions"],
                    "Total_Runtime": solution["total_runtime"],
                }
            )

        # Create DataFrame and export
        df = pd.DataFrame(export_data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pareto_solutions_{self.p_median}p_{timestamp}.csv"

        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(solutions_dict)} Pareto solutions to {filename}")

        return filename
