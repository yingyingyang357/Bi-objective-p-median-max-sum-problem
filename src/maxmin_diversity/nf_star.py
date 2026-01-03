import time
import logging
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import Optional, Tuple, ClassVar, Set, Dict, List
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData
from src.maxmin_diversity.nf import NewCompactFormulation

# configure logger for the NF* method
logger = logging.getLogger(__name__)


class NFwithBounds(OptimisationModelBase):
    """
    The NF* method for p-dispersion optimisation. The algorithm follows the
    reduced new compact formulation with lower and upper bounds (NF*) as in:

        Sayah, D., Irnich, S., 2017.
        A new compact formulation for the discrete p-dispersion problem.
        European Journal of Operational Research 256, 62–67.

    """

    name: ClassVar[str] = "NF* Method for P-Dispersion"
    time_limit: float = 3600  # default maximum solving time in seconds

    def __init__(self, **data):
        super().__init__(**data)
        # allow overriding through kwargs while retaining defaults
        self.time_limit = data.get("time_limit", self.time_limit)

    # ------------------------------
    # Bounds (UB and LB)
    # ------------------------------
    @staticmethod
    def find_upper_bound(n: int, p: int, q: np.ndarray) -> Optional[float]:
        """
        Upper bound by:
        - For each i, sort distances to others descending, take (p-1)-th largest.
        - Then take the p-th largest among these.
        """
        if p <= 0 or p > n or q is None or q.shape != (n, n):
            return None
        try:
            d_p_minus_1: List[float] = []
            for i in range(n):
                row = [float(q[i, j]) for j in range(n) if j != i]
                if len(row) < (p - 1):
                    # Not enough neighbours to take (p-1)-th largest
                    return None
                row_sorted_desc = sorted(row, reverse=True)
                d_p_minus_1.append(row_sorted_desc[p - 2])  # index p-2 (0-based)
            d_sorted_desc = sorted(d_p_minus_1, reverse=True)
            if len(d_sorted_desc) < p:
                return None
            return float(d_sorted_desc[p - 1])  # index p-1 (0-based)
        except Exception:
            return None

    @staticmethod
    def find_lower_bound(n: int, p: int, q: np.ndarray, D_list: np.ndarray, max_time: float) -> Optional[float]:
        """
        Lower bound via greedy independent-set construction on graphs G(Dk):
          - Vertices: {0..n-1}, edge (i,j) if d[i,j] < Dk.
          - Greedy: repeatedly pick vertex of minimum current degree; remove it and its neighbours.
          - If the independent set size >= p, then LB = Dk.
          - Iterate Dk in decreasing order until success or timeout.
        """
        if p <= 0 or p > n or q is None or q.shape != (n, n):# or D_list.size == 0:
            return None

        start = time.time()
        for Dk in sorted(D_list, reverse=True):
            # Build adjacency where edges denote distances strictly less than Dk
            adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(i + 1, n):
                    if q[i, j] < Dk:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

            # Greedy independent set
            remaining = set(range(n))
            indep: Set[int] = set()

            while remaining:
                if (time.time() - start) >= max_time:
                    logger.warning("Max time reached during LB construction. Exiting early.")
                    return None
                # vertex of minimum degree in the induced subgraph
                min_deg_node = min(remaining, key=lambda v: len(adjacency[v] & remaining))
                indep.add(min_deg_node)
                neighbours = adjacency[min_deg_node] & remaining
                remaining -= neighbours
                if min_deg_node in remaining:
                    remaining.remove(min_deg_node)

            if len(indep) >= p:
                return float(Dk)

        return None

    # ------------------------------
    # Core NF* model build & solve 
    # ------------------------------
    def nf_with_bounds(
        self,
        n: int,
        p_median: int,
        q: np.ndarray,
        time_limit: float,
    ) -> Tuple[Optional[float], float, Set[str]]:
        """
        Build and solve NF compact model with index restriction inferred from bounds.
        Returns only (opt_value, runtime, warnings) as requested.
        """
        warnings: Set[str] = set()
        start_time = time.time()

        # Basic checks
        if n <= 0 or p_median <= 0 or q is None or q.shape != (n, n):
            warnings.add("Invalid inputs to NF* model.")
            return None, time.time() - start_time, warnings

        # Prepare D_list
        D_list = NewCompactFormulation.unique_sorted_upper_triangle_distances(q)
        if D_list.size == 0:
            warnings.add("No distances found in upper triangle.")
            return None, time.time() - start_time, warnings

        # find the upper bound and lower bound for the optimal max-min value 
        upperb = self.find_upper_bound(n,p_median,q)
        lowerb = self.find_lower_bound(n,p_median,q,D_list,time_limit)
        
        # Find indices corresponding to lb and ub in D_list
        k_min = next(k for k, val in enumerate(D_list) if val == lowerb)
        k_max = next(k for k, val in enumerate(D_list) if val == upperb)

        # Clip to valid range
        k_min = max(0, min(k_min, len(D_list) - 1))
        k_max = max(0, min(k_max, len(D_list) - 1))

        if k_min > k_max:
            warnings.add("Resolved k_min > k_max; resetting to full range.")
            k_min, k_max = 0, len(D_list) - 1

        # Obtain the number of all different non-zero values in qij, here K_bar is K_0 in paper 
        K_bar = range(len(D_list))
        
        # Define the set E(D^k)={(i,j) in E: d[i,j] < D^k}
        E = {(i, j) for i in range(n) for j in range(n) if i < j}
        E_D = dict()  # to store E(D[k]) for each k
        for k in K_bar:  # skip k = 0
            E_D[k] = {(i, j) for (i, j) in E if q[i, j] < D_list[k]}   

        # Redefine K as {k_min, ..., k_max}
        K = range(k_min, k_max + 1)  
        E_D[K[0] - 1] =  E_D[0] 

        # Build model
        m = Model(name="NF*_Reduced_P-Dispersion")

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
        m.maximize(D_list[K[0]-1] + sum((D_list[k]-D_list[k-1])*z[k] for k in K))
        m.parameters.timelimit = time_limit

        try:
            solution = m.solve()
        except Exception as e:
            msg = f"NF* optimisation failed with exception: {str(e)}"
            warnings.add(msg)
            logger.error(msg, exc_info=True)
            return None, time.time() - start_time, warnings

        runtime = time.time() - start_time

        if solution is None:
            msg = "NF* restricted model returned no solution (infeasible or solver error)."
            warnings.add(msg)
            logger.error(msg)
            return None, runtime, warnings

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
        The main optimise function of the NF* method for p-dispersion problems.

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

        # Validations (aligned with your DBS/NF approach)
        if validation:
            if getattr(problem, "number_of_facilities", 0) <= 0:
                msg = f"Invalid number of facilities: {getattr(problem, 'number_of_facilities', None)}"
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

        n = problem.number_of_facilities
        q = problem.distance_matrix

        logger.info(f"Starting NF* optimisation for {n} facilities, p={p_median}")

        # Execute NF* with minimal-return solver
        try:
            opt_value, runtime, nf_warnings = self.nf_with_bounds(
                n=n,
                p_median=p_median,
                q=q,
                time_limit=self.time_limit,
            )
            optimisation_warnings.update(nf_warnings)

            # Prepare results DataFrame (Curtin-style)
            if opt_value is not None:
                status = "optimal" if len(optimisation_warnings) == 0 else "suboptimal"
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
                    f"NF* completed: distance={opt_value:.6f}, runtime_seconds={runtime:.3f}s"
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
                msg = f"NF* failed to find solution for p={p_median}"
                optimisation_warnings.add(msg)
                logger.error(msg)

        except Exception as e:
            err = f"NF* optimisation failed with exception: {str(e)}"
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


