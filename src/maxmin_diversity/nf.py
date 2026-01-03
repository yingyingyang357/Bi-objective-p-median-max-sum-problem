import time
import logging
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import Optional, Tuple, ClassVar, Set, Dict
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData

# configure logger for the NF method
logger = logging.getLogger(__name__)

class NewCompactFormulation(OptimisationModelBase):
    """
    The NF method for p-dispersion optimisation. The Algorithm is implemented as in
    the new compact formulation (NF) algorithm from:

        Sayah, D., Irnich, S., 2017.
        A new compact formulation for the discrete p-dispersion problem.
        European Journal of Operational Research 256, 62–67.

    This class builds the NF compact model and solves it in one pass, returning the
    max-min objective value (the guaranteed minimum distance among chosen facilities).
    """

    name: ClassVar[str] = "NF Method for P-Dispersion"
    time_limit: float = 3600  # default maximum solving time in seconds

    def __init__(self, **data):
        super().__init__(**data)
        # allow overriding through kwargs while retaining defaults
        self.time_limit = data.get("time_limit", self.time_limit)

    # ------------------------------
    # Helper construction utilities
    # ------------------------------
    @staticmethod
    def unique_sorted_upper_triangle_distances(distance_matrix: np.ndarray) -> np.ndarray:
        """
        Extracts the sorted unique (non-negative) distances from the upper triangle (excluding diagonal).
        This yields the vector D_list used by the NF formulation.
        """
        # Guard against non-square or empty input
        if distance_matrix is None or distance_matrix.size == 0:
            return np.array([], dtype=float)

        # Upper-triangular indices excluding diagonal
        triu_i, triu_j = np.triu_indices_from(distance_matrix, k=1)
        upper_values = distance_matrix[triu_i, triu_j].astype(float)

        unique_sorted = np.unique(upper_values)
        return unique_sorted

    @staticmethod
    def build_E_sets(distance_matrix: np.ndarray, D_list: np.ndarray) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Builds E(D^k) = { (i, j) in E : d[i, j] < D^k } for all k in {0, ..., k_max},
        where E = { (i, j) : i < j }.

        Returns a dict: k -> set of pairs (i, j).
        """
        n = distance_matrix.shape[0]
        E = {(i, j) for i in range(n) for j in range(i + 1, n)}
        E_D: Dict[int, Set[Tuple[int, int]]] = {}

        # For each threshold D^k, collect pairs whose distance is strictly less than D^k
        for k, Dk in enumerate(D_list):
            E_D[k] = {(i, j) for (i, j) in E if distance_matrix[i, j] < Dk}
        return E_D

    # ------------------------------
    # Core NF model build & solve
    # ------------------------------
    def new_compact_formulation(
        self,
        n: int,
        p_median: int,
        q: np.ndarray,
        time_limit: float,
    ) -> Tuple[Optional[float], float, Set[str]]:
        """
        Internal method that builds and solves the NF compact model.

        Returns:
            (opt_value, runtime, warnings)
        """
        warnings: Set[str] = set()
        start_time = time.time()

        # Validate inputs
        if n <= 0:
            msg = f"Invalid number of facilities: {n}"
            warnings.add(msg)
            logger.error(msg)
            return None, [], time.time() - start_time, 0.0, warnings

        if p_median <= 0:
            msg = f"Invalid p_median value: {p_median}. Must be positive."
            warnings.add(msg)
            logger.error(msg)
            return None, [], time.time() - start_time, 0.0, warnings

        if q is None or q.shape != (n, n):
            msg = "Distance matrix is None or not square. Cannot perform optimisation."
            warnings.add(msg)
            logger.error(msg)
            return None, [], time.time() - start_time, 0.0, warnings

        if time_limit <= 0:
            msg = f"Invalid time limit: {time_limit}. Using default 3600 seconds."
            warnings.add(msg)
            logger.warning(msg)
            time_limit = 3600

        # Construct D_list (sorted unique upper-triangle distances)
        D_list = self.unique_sorted_upper_triangle_distances(q)

        if D_list.size == 0:
            msg = "No distance values found in upper triangle."
            warnings.add(msg)
            logger.error(msg)
            return None, [], time.time() - start_time, 0.0, warnings

        # K_bar = {0, ..., k_max}; K = K_bar \ {0}
        K_bar = list(range(len(D_list)))
        if len(K_bar) < 2:
            # If there is only one unique value, the model still builds,
            # but z-variables will be empty and the objective becomes D_list[0].
            logger.warning("Only one unique distance found; NF objective degenerates to the single value.")
        K = K_bar[1:]  # skip k = 0

        # Build E(D^k) sets
        E_D = self.build_E_sets(q, D_list)

        # Build model
        m = Model(name="NF_Compact_P-Dispersion")
        
        # Decision variables
        xi = [i for i in range(n)]
        x = m.binary_var_dict(keys=xi, name="x") 
        zk = [k for k in K]
        z = m.binary_var_dict(keys=zk, name="z") 
        
        # Constraints
        [m.add_constraint(m.sum([x[i] for i in range(n)]) == p_median)]
        for k in K[1:]:
            m.add_constraint(z[k] <= z[k-1])
        
        for k in K:
            for (i, j) in E_D[k] - E_D[k - 1]: 
                m.add_constraint(x[i]+x[j]+z[k] <= 2) 
            
        # objective function
        m.maximize(D_list[0] + sum((D_list[k]-D_list[k-1])*z[k] for k in K))
        m.parameters.timelimit = time_limit

        # Solve
        try:
            solution = m.solve()
        except Exception as e:
            msg = f"NF optimisation failed with exception: {str(e)}"
            warnings.add(msg)
            logger.error(msg, exc_info=True)
            return None, [], time.time() - start_time, 0.0, warnings

        # runtime_solver = float(m.solve_details.time) if m.solve_details is not None else 0.0
        runtime = time.time() - start_time

        if solution is None:
            msg = "NF compact model returned no solution (infeasible or solver error)."
            warnings.add(msg)
            logger.error(msg)
            return None, [], runtime, warnings

        # Extract objective and selected facilities
        try:
            opt_value = float(m.objective_value)
        except Exception:
            opt_value = None
            warnings.add("Objective value could not be retrieved.")
            logger.warning("Objective value could not be retrieved.")


        return opt_value, runtime, warnings

    # ------------------------------
    # Public API
    # ------------------------------
    def optimise(
        self,
        problem: ProblemData,
        p_median: int,
        validation: bool = False
    ) -> Tuple[Set[str], pd.DataFrame]:
        """
        The main optimise function of the NF method for p-dispersion problems.

        Args:
            problem (ProblemData): Contains distance matrix and number of facilities.
            p_median (int): Number of facilities to select.
            validation (bool): Whether to perform input validation before optimisation.

        Returns:
            A tuple containing:
                Set[str]: A set of warning messages produced by the optimisation model.
                pd.DataFrame: A dataframe summarising the optimisation outcome.
        """
        optimisation_warnings: Set[str] = set()

        # Basic validations (similar style to your DBS class)
        if validation:
            if getattr(problem, "number_of_facilities", 0) <= 0:
                msg = f"Invalid number of facilities: {problem.number_of_facilities}"
                optimisation_warnings.add(msg)
                logger.error(msg)
                return optimisation_warnings, pd.DataFrame()

            if getattr(problem, "distance_matrix", None) is None:
                msg = "Distance matrix is None. Cannot perform optimisation."
                optimisation_warnings.add(msg)
                logger.error(msg)
                return optimisation_warnings, pd.DataFrame()

            if p_median <= 0:
                msg = f"Invalid p_median: {p_median}. Must be positive."
                optimisation_warnings.add(msg)
                logger.error(msg)
                return optimisation_warnings, pd.DataFrame()

            if p_median > problem.number_of_facilities:
                msg = (
                    f"p_median ({p_median}) exceeds number of facilities "
                    f"({problem.number_of_facilities})."
                )
                optimisation_warnings.add(msg)
                logger.error(msg)
                return optimisation_warnings, pd.DataFrame()

            if self.time_limit <= 0:
                msg = f"Invalid time limit: {self.time_limit}. Using default 3600 seconds."
                optimisation_warnings.add(msg)
                logger.warning(msg)
                self.time_limit = 3600

        logger.info(
            f"Starting NF optimisation for {problem.number_of_facilities} facilities, p={p_median}"
        )

        # Execute NF
        try:
            opt_value, runtime, nf_warnings = self.new_compact_formulation(
                n=problem.number_of_facilities,
                p_median=p_median,
                q=problem.distance_matrix,
                time_limit=self.time_limit,
            )

            optimisation_warnings.update(nf_warnings)

            # Prepare results DataFrame (aligned with your DBS summary style)
            if opt_value is not None:
                status = "optimal" if len(nf_warnings) == 0 else "suboptimal"
                results_df = pd.DataFrame(
                    {
                        "algorithm": [self.name],
                        "p_median": [p_median],
                        "optimal_distance": [opt_value],
                        "runtime_seconds": [runtime],
                        "time_limit": [self.time_limit],
                        "status": [status],
                    }
                )
                logger.info(
                    f"NF completed: distance={opt_value:.6f}, "
                    f"runtime_seconds={runtime:.3f}s, "
                )
            else:
                results_df = pd.DataFrame(
                    {
                        "algorithm": [self.name],
                        "p_median": [p_median],
                        "optimal_distance": [None],
                        "runtime_seconds": [runtime],
                        "time_limit": [self.time_limit],
                        "status": ["failed"],
                    }
                )
                msg = f"NF failed to find solution for p={p_median}"
                optimisation_warnings.add(msg)
                logger.error(msg)

        except Exception as e:
            err = f"NF optimisation failed with exception: {str(e)}"
            optimisation_warnings.add(err)
            logger.error(err, exc_info=True)

            results_df = pd.DataFrame(
                {
                    "algorithm": [self.name],
                    "p_median": [p_median],
                    "optimal_distance": [None],
                    "runtime_seconds": [0.0],
                    "time_limit": [self.time_limit],
                    "status": ["error"],
                }
            )

        if len(optimisation_warnings) > 0:
            logger.warning(f"Optimisation completed with {len(optimisation_warnings)} warning(s)")

        return optimisation_warnings, results_df
