import time
import logging
import pandas as pd
import numpy as np
from typing import ClassVar, Dict, Any
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData
from src.p_median.bender_decomposition import BendersDecompositionModel
from src.p_median.beta_model import BetaPmeanModel

# configure logger for the binary search module
logger = logging.getLogger(__name__)


class PMedianOptimiser(OptimisationModelBase):
    """
    The P-Median optimiser class that extends the OptimisationModelBase.
    This class implements the p-median problem optimisation with dispersion constraints
    using various methods, including
    1. Branch and Cut with Benders Decomposition
    2. Cutting Plane Method
    3. Standard MIP Solver
    4. Standard Benders Decomposition Method
    """

    name: ClassVar[str] = "P-Median Optimiser"

    def __init__(self, **data):
        # Only call super().__init__() without arguments as OptimisationModelBase expects
        super().__init__()
        # Store any additional data parameters for potential future use
        self._config_data = data

    def solve_standard_p_median(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float = 0.0,
        time_limit: int = 3600,
        validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve standard p-median problem using direct MIP formulation.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            time_limit: Maximum computation time in seconds
            validation: Whether to perform input validation

        Returns:
            Dict containing solution results and statistics
        """
        start_time = time.time()
        warnings = set()

        try:
            # Create standard p-median model
            beta_model = BetaPmeanModel(
                p_median=p_median,
                min_distance=min_distance,
                problem_data=problem,
                validate=validation,
            )

            # Solve standard p-median (no dispersion constraint)
            logger.info(f"Solving standard p-median with p={p_median}")

            solution = beta_model.p_median_model.solve()
            runtime = time.time() - start_time

            if solution is None:
                return {
                    "status": "infeasible",
                    "runtime": runtime,
                    "warnings": warnings,
                    "solution": None,
                    "objective_value": None,
                    "selected_facilities": None,
                }

            # Extract solution
            x_solution = np.array(solution.get_value_list(beta_model.x))
            objective_value = solution.get_objective_value()
            selected_facilities = np.where(x_solution > 0.5)[0].tolist()

            logger.info(
                f"Standard p-median solved in {runtime:.2f}s, objective: {objective_value:.6f}"
            )

            return {
                "status": "optimal",
                "runtime": runtime,
                "warnings": warnings,
                "solution": x_solution,
                "objective_value": objective_value,
                "selected_facilities": selected_facilities,
                "method": "standard_p_median",
            }

        except Exception as e:
            logger.error(f"Error in standard p-median solution: {e}")
            return {
                "status": "error",
                "runtime": time.time() - start_time,
                "warnings": warnings.union({str(e)}),
                "solution": None,
                "objective_value": None,
                "selected_facilities": None,
            }

    def solve_quadratic_model(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float,
        beta: float,
        epsilon: float = 1,
        current_time: float = 0.0,
        validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve p-median with dispersion constraint using quadratic formulation.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            beta: Minimum required dispersion value
            epsilon: Tolerance for dispersion constraint
            time_limit: Maximum computation time
            validation: Whether to perform validation

        Returns:
            Dict containing solution results and statistics
        """
        start_time = time.time()
        warnings = set()

        try:
            # Create beta model
            beta_model = BetaPmeanModel(
                p_median=p_median,
                min_distance=min_distance,
                problem_data=problem,
                validate=validation,
            )

            logger.info(
                f"Solving quadratic model with beta={beta:.6f}, epsilon={epsilon:.6f}"
            )

            # Solve with quadratic dispersion constraint
            result = beta_model.quadratic_model(
                beta=beta, epsilon=epsilon, current_time=current_time
            )
            runtime = time.time() - start_time

            if isinstance(result, str):
                return {
                    "status": "infeasible",
                    "runtime": runtime,
                    "warnings": warnings.union({result}),
                    "solution": None,
                    "objective_value": None,
                    "selected_facilities": None,
                    "achieved_dispersion": None,
                }

            (
                solution_vector,
                objective_value,
                solve_time,
                achieved_dispersion,
                status,
            ) = result
            selected_facilities = np.where(solution_vector > 0.5)[0].tolist()

            logger.info(
                f"Quadratic model solved in {runtime:.2f}s, objective: {objective_value:.6f}"
            )

            return {
                "status": status,
                "runtime": runtime,
                "warnings": warnings,
                "solution": solution_vector,
                "objective_value": objective_value,
                "selected_facilities": selected_facilities,
                "achieved_dispersion": achieved_dispersion,
                "method": "quadratic_model",
            }

        except Exception as e:
            logger.error(f"Error in quadratic model solution: {e}")
            return {
                "status": "error",
                "runtime": time.time() - start_time,
                "warnings": warnings.union({str(e)}),
                "solution": None,
                "objective_value": None,
                "selected_facilities": None,
                "achieved_dispersion": None,
            }

    def solve_cutting_plane_model(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float,
        beta: float,
        epsilon: float = 1,
        max_iterations: int = 10**9,
        current_time: float = 0,
        validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve p-median with dispersion using cutting plane method.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            beta: Minimum required dispersion value
            epsilon: Tolerance for dispersion constraint
            max_iterations: Maximum cutting plane iterations
            time_limit: Maximum computation time
            validation: Whether to perform validation

        Returns:
            Dict containing solution results and statistics
        """
        start_time = time.time()
        warnings = set()

        try:

            # Create beta model
            beta_model = BetaPmeanModel(
                p_median=p_median,
                min_distance=min_distance,
                problem_data=problem,
                validate=validation,
            )

            logger.info(
                f"Solving cutting plane model with beta={beta:.6f}, max_iter={max_iterations}"
            )

            # Solve with cutting plane method
            result = beta_model.cutting_plane_model(
                beta=beta,
                epsilon=epsilon,
                max_iterations=max_iterations,
                current_time=current_time,
            )

            runtime = time.time() - start_time

            if isinstance(result, str):
                return {
                    "status": "infeasible",
                    "runtime": runtime,
                    "warnings": warnings.union({result}),
                    "solution": None,
                    "objective_value": None,
                    "selected_facilities": None,
                    "achieved_dispersion": None,
                }
            (
                solution_vector,
                objective_value,
                solve_time,
                achieved_dispersion,
                status,
            ) = result
            selected_facilities = np.where(solution_vector > 0.5)[0].tolist()

            logger.info(
                f"Cutting plane model solved in {runtime:.2f}s, objective: {objective_value:.6f}"
            )

            return {
                "status": status,
                "runtime": runtime,
                "warnings": warnings,
                "solution": solution_vector,
                "objective_value": objective_value,
                "selected_facilities": selected_facilities,
                "achieved_dispersion": achieved_dispersion,
                "iterations": max_iterations,
                "method": "cutting_plane",
            }

        except Exception as e:
            logger.error(f"Error in cutting plane solution: {e}")
            return {
                "status": "error",
                "runtime": time.time() - start_time,
                "warnings": warnings.union({str(e)}),
                "solution": None,
                "objective_value": None,
                "selected_facilities": None,
                "achieved_dispersion": None,
            }

    def solve_benders_decomposition(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float,
        beta: float,
        epsilon: float = 1,
        max_iterations: int = 10**9,
        current_time: float = 0.0,
        validation: bool = False,
        cutting_plane: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve p-median with dispersion using classical Benders decomposition.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            beta: Minimum required dispersion value
            epsilon: Tolerance for dispersion constraint
            max_iterations: Maximum Benders iterations
            time_limit: Maximum computation time
            validation: Whether to perform validation

        Returns:
            Dict containing solution results and statistics
        """
        start_time = time.time()
        warnings = set()

        try:
            # Create Benders decomposition model
            benders_model = BendersDecompositionModel(
                p_median=p_median,
                min_distance=min_distance,
                problem_data=problem,
                validate=validation,
            )

            logger.info(f"Solving Benders decomposition with beta={beta:.6f}")

            # Solve using Balinski cuts (classical Benders)
            result = benders_model.balinski_cut(
                beta=beta,
                epsilon=epsilon,
                cutting_plane=cutting_plane,
                current_time=current_time,
                cutting_plane_iter=max_iterations,
            )

            runtime = time.time() - start_time

            if isinstance(result, str):
                return {
                    "status": "infeasible",
                    "runtime": runtime,
                    "warnings": warnings.union({result}),
                    "solution": None,
                    "objective_value": None,
                    "selected_facilities": None,
                    "achieved_dispersion": None,
                }

            (
                solution_vector,
                objective_value,
                solve_time,
                achieved_dispersion,
                status,
            ) = result
            selected_facilities = np.where(solution_vector > 0.5)[0].tolist()

            logger.info(
                f"Benders decomposition solved in {runtime:.2f}s, objective: {objective_value:.6f}"
            )

            return {
                "status": status,
                "runtime": runtime,
                "warnings": warnings,
                "solution": solution_vector,
                "objective_value": objective_value,
                "selected_facilities": selected_facilities,
                "achieved_dispersion": achieved_dispersion,
                "max_iterations": max_iterations,
                "method": "benders_decomposition",
            }

        except Exception as e:
            logger.error(f"Error in Benders decomposition: {e}")
            return {
                "status": "error",
                "runtime": time.time() - start_time,
                "warnings": warnings.union({str(e)}),
                "solution": None,
                "objective_value": None,
                "selected_facilities": None,
                "achieved_dispersion": None,
            }

    def solve_branch_and_cut_benders(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float,
        beta: float,
        epsilon: float = 1,
        current_time: int = 3600,
        cutting_plane: bool = True,
        validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve p-median with dispersion using branch-and-cut with Benders lazy constraints.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            beta: Minimum required dispersion value
            epsilon: Tolerance for dispersion constraint
            time_limit: Maximum computation time
            cutting_plane: Whether to add cutting plane constraints
            validation: Whether to perform validation

        Returns:
            Dict containing solution results and statistics
        """
        start_time = time.time()
        warnings = set()

        try:
            # Create Benders decomposition model
            benders_model = BendersDecompositionModel(
                p_median=p_median,
                min_distance=min_distance,
                problem_data=problem,
                validate=validation,
            )

            logger.info(f"Solving branch-and-cut Benders with beta={beta:.6f}")

            # Solve using branch-and-cut with lazy constraints
            result = benders_model.branch_and_cut_benders(
                beta=beta,
                epsilon=epsilon,
                cutting_plane=cutting_plane,
                current_time=current_time,
            )
            runtime = time.time() - start_time

            if isinstance(result, str):
                return {
                    "status": "infeasible",
                    "runtime": runtime,
                    "warnings": warnings.union({result}),
                    "solution": None,
                    "objective_value": None,
                    "selected_facilities": None,
                    "achieved_dispersion": None,
                }

            (
                solution_vector,
                lower_bound,
                upper_bound,
                solve_time,
                achieved_dispersion,
            ) = result
            selected_facilities = np.where(solution_vector > 0.5)[0].tolist()

            # Calculate optimality gap
            gap = (
                (upper_bound - lower_bound) / abs(upper_bound) * 100
                if abs(upper_bound) > 1e-6
                else 0.0
            )

            logger.info(
                f"Branch-and-cut Benders solved in {runtime:.2f}s, objective: {lower_bound:.6f}"
            )

            return {
                "status": "optimal",
                "runtime": runtime,
                "warnings": warnings,
                "solution": solution_vector,
                "objective_value": lower_bound,
                "upper_bound": upper_bound,
                "optimality_gap": gap,
                "selected_facilities": selected_facilities,
                "achieved_dispersion": achieved_dispersion,
                "method": "branch_and_cut_benders",
            }

        except Exception as e:
            logger.error(f"Error in branch-and-cut Benders: {e}")
            return {
                "status": "error",
                "runtime": time.time() - start_time,
                "warnings": warnings.union({str(e)}),
                "solution": None,
                "objective_value": None,
                "selected_facilities": None,
                "achieved_dispersion": None,
            }

    def solve_all_methods(
        self,
        problem: ProblemData,
        p_median: int,
        min_distance: float,
        beta: float,
        epsilon: float = 1,
        time_limit: int = 3600,
        validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve p-median problem using all available methods and compare results.

        Args:
            problem: Problem instance data
            p_median: Number of facilities to select
            min_distance: Minimum distance between selected facilities
            beta: Minimum required dispersion value
            epsilon: Tolerance for dispersion constraint
            time_limit: Maximum computation time per method
            validation: Whether to perform validation

        Returns:
            Dict containing results from all methods and comparison
        """
        start_time = time.time()
        results = {}

        logger.info(f"Solving p-median with all methods: p={p_median}, beta={beta:.6f}")

        # Method 1: Standard P-Median (baseline)
        try:
            results["standard"] = self.solve_standard_p_median(
                problem, p_median, min_distance, time_limit, validation
            )
        except Exception as e:
            results["standard"] = {"status": "error", "error": str(e)}

        # Method 2: Quadratic Model
        try:
            results["quadratic"] = self.solve_quadratic_model(
                problem, p_median, min_distance, beta, epsilon, time_limit, validation
            )
        except Exception as e:
            results["quadratic"] = {"status": "error", "error": str(e)}

        # Method 3: Cutting Plane
        try:
            results["cutting_plane"] = self.solve_cutting_plane_model(
                problem,
                p_median,
                min_distance,
                beta,
                epsilon,
                10**9,
                time_limit,
                validation,
            )
        except Exception as e:
            results["cutting_plane"] = {"status": "error", "error": str(e)}

        # Method 4: Benders Decomposition
        try:
            results["benders"] = self.solve_benders_decomposition(
                problem,
                p_median,
                min_distance,
                beta,
                epsilon,
                10**9,
                time_limit,
                validation,
            )
        except Exception as e:
            results["benders"] = {"status": "error", "error": str(e)}

        # Method 5: Branch-and-Cut Benders
        try:
            results["branch_cut_benders"] = self.solve_branch_and_cut_benders(
                problem,
                p_median,
                min_distance,
                beta,
                epsilon,
                time_limit,
                True,
                validation,
            )
        except Exception as e:
            results["branch_cut_benders"] = {"status": "error", "error": str(e)}

        total_runtime = time.time() - start_time

        # Create comparison summary
        comparison = self._create_comparison_summary(results)

        logger.info(f"All methods completed in {total_runtime:.2f}s")

        return {
            "results": results,
            "comparison": comparison,
            "total_runtime": total_runtime,
            "problem_info": {
                "p_median": p_median,
                "min_distance": min_distance,
                "beta": beta,
                "epsilon": epsilon,
                "n_facilities": problem.number_of_facilities,
            },
        }

    def _create_comparison_summary(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a comparison summary of all solution methods.

        Args:
            results: Dictionary containing results from all methods

        Returns:
            DataFrame with comparison of all methods
        """
        comparison_data = []

        for method_name, result in results.items():
            if isinstance(result, dict) and result.get("status") != "error":
                comparison_data.append(
                    {
                        "Method": method_name,
                        "Status": result.get("status", "unknown"),
                        "Runtime (s)": round(result.get("runtime", 0), 2),
                        "Objective Value": (
                            round(result.get("objective_value", 0), 6)
                            if result.get("objective_value") is not None
                            else None
                        ),
                        "Achieved Dispersion": (
                            round(result.get("achieved_dispersion", 0), 6)
                            if result.get("achieved_dispersion") is not None
                            else None
                        ),
                        "Selected Facilities": len(
                            result.get("selected_facilities", [])
                        ),
                        "Gap (%)": (
                            round(result.get("optimality_gap", 0), 2)
                            if result.get("optimality_gap") is not None
                            else None
                        ),
                    }
                )
            else:
                comparison_data.append(
                    {
                        "Method": method_name,
                        "Status": "error",
                        "Runtime (s)": 0,
                        "Objective Value": None,
                        "Achieved Dispersion": None,
                        "Selected Facilities": 0,
                        "Gap (%)": None,
                    }
                )

        return pd.DataFrame(comparison_data)
