import time
import logging
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from typing import Optional, Tuple, ClassVar, Set, Dict, List
import networkx as nx
from src.solver_class import OptimisationModelBase
from src.data_class import ProblemData
from src.maxmin_diversity.nf import NewCompactFormulation
from src.maxmin_diversity.nf_star import NFwithBounds

# configure logger for the NFC* method
logger = logging.getLogger(__name__)


class NFwithBoundsCuts(OptimisationModelBase):
    """
    The NFC* method for p-dispersion optimisation. The algorithm follows the
    reduced new compact formulation with lower and upper bounds and valid inequalities (NFC*) as in:

        Sayah, D., Irnich, S., 2017.
        A new compact formulation for the discrete p-dispersion problem.
        European Journal of Operational Research 256, 62–67.

    """

    name: ClassVar[str] = "NFC* Method for P-Dispersion"
    time_limit: float = 3600  # default maximum solving time in seconds

    def __init__(self, **data):
        super().__init__(**data)
        # allow overriding through kwargs while retaining defaults
        self.time_limit = data.get("time_limit", self.time_limit)

    # ------------------------------
    # Valid inequalities (Section 2.3.2): helper functions
    # ------------------------------
    @staticmethod
    def greedy_max_weight_clique(n_range, E_Dk, x_vals, z_k_val) -> List[int]:
        """
        Approximate max weight clique S in G = (I, E_Dk), using greedy heuristic.
        w_i = x̄_i + z̄_k - 1 based on fractional relaxation values.
        """
        G = nx.Graph()
        G.add_nodes_from(n_range)
        G.add_edges_from(E_Dk)

        weights = {i: x_vals[i] + z_k_val - 1 for i in n_range}

        S: List[int] = []
        while G.nodes:
            scores = {node: G.degree[node] * weights[node] for node in G.nodes}
            if not scores:
                break
            best_node = max(scores, key=scores.get)
            S.append(best_node)
            # keep only neighbours to maintain a clique
            neighbours = set(G.neighbors(best_node))
            G = G.subgraph(neighbours).copy()

        return S

    @staticmethod
    def find_violated_inequalities(
        n: int,
        K,
        E: Set[Tuple[int, int]],
        E_D: Dict[int, Set[Tuple[int, int]]],
        x_vals: Dict[int, float],
        z_vals: Dict[int, float],
    ) -> List[Tuple[int, List[int]]]:
        """
        For each k in K, apply greedy separation and return violated inequalities.
        """
        n_range = range(n)
        violated: List[Tuple[int, List[int]]] = []
        for k in K:
            z_k_val = z_vals[k]
            E_Dk = E_D[k]

            S = NFwithBoundsCuts.greedy_max_weight_clique(n_range, E_Dk, x_vals, z_k_val)
            sum_x = sum(x_vals[i] for i in S)
            lhs = sum_x + (len(S) - 1) * z_k_val
            rhs = len(S)

            if lhs > rhs:
                violated.append((k, S))

        return violated

    # ------------------------------
    # Relaxed NF compact model (for fractional x,z used in separation)
    # ------------------------------
    def _max_min_compact_relaxed(
        self,
        n: int,
        p_median: int,
        q: np.ndarray,
        time_limit: float,
    ) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
        """
        Solve the linear relaxation of the NF compact model (continuous x,z in [0,1]).
        Returns (objective_value, runtime, x_vals, z_vals).
        """
        start = time.time()

        # D_list and sets
        D_list = NewCompactFormulation.unique_sorted_upper_triangle_distances(q)
        K_bar = range(len(D_list))
        K = K_bar[1:]  # skip k=0

        E = {(i, j) for i in range(n) for j in range(n) if i < j}
        E_D: Dict[int, Set[Tuple[int, int]]] = {k: {(i, j) for (i, j) in E if q[i, j] < D_list[k]} for k in K_bar}

        # Relaxed compact model
        m = Model(name="MAX_MIN_Relaxed")
        xi = [i for i in range(n)]
        x = m.continuous_var_dict(keys=xi, name="x", lb=0, ub=1)
        zk = [k for k in K]
        z = m.continuous_var_dict(keys=zk, name="z", lb=0, ub=1)

        [m.add_constraint(m.sum([x[i] for i in range(n)]) == p_median)]
        for k in K[1:]:
            m.add_constraint(z[k] <= z[k - 1])

        for k in K:
            for (i, j) in E_D[k] - E_D[k - 1]:
                m.add_constraint(x[i] + x[j] + z[k] <= 2)

        m.maximize(D_list[0] + sum((D_list[k] - D_list[k - 1]) * z[k] for k in K))
        m.parameters.timelimit = time_limit

        m.solve()
        sol = m.solution
        obj = m.objective_value

        x_vals = {i: sol.get_value(x[i]) for i in range(n)}
        z_vals = {k: sol.get_value(z[k]) for k in K}

        runtime = time.time() - start
        return obj, runtime, x_vals, z_vals

    # ------------------------------
    # Core NFC* solve: bounds + cuts
    # ------------------------------
    def nf_with_bounds_cuts(
        self,
        n: int,
        p_median: int,
        q: np.ndarray,
        time_limit: float,
    ) -> Tuple[Optional[float], float, Set[str]]:
        """
        Build and solve NF compact model restricted by bounds and augmented
        with valid clique inequalities. Returns (opt_value, runtime, warnings).
        """
        warnings: Set[str] = set()
        start_time = time.time()

        # Basic checks
        if n <= 0 or p_median <= 0 or q is None or q.shape != (n, n):
            warnings.add("Invalid inputs to NFC* model.")
            return None, time.time() - start_time, warnings

        # Prepare D_list
        D_list = NewCompactFormulation.unique_sorted_upper_triangle_distances(q)
        if D_list.size == 0:
            warnings.add("No distance values found in upper triangle.")
            return None, time.time() - start_time, warnings

        # E and E(D^k)
        K_bar = range(len(D_list))
        E = {(i, j) for i in range(n) for j in range(n) if i < j}
        E_D: Dict[int, Set[Tuple[int, int]]] = {k: {(i, j) for (i, j) in E if q[i, j] < D_list[k]} for k in K_bar}

        # Bounds
        upperb = NFwithBounds.find_upper_bound(n, p_median, q)
        lowerb = NFwithBounds.find_lower_bound(n, p_median, q, D_list, time_limit)

        # Map bounds to indices (exact match as per your logic)
        try:
            k_min = next(k for k, val in enumerate(D_list) if val == lowerb)
        except StopIteration:
            warnings.add("Lower bound not in D_list; defaulting k_min=0.")
            k_min = 0
        try:
            k_max = next(k for k, val in enumerate(D_list) if val == upperb)
        except StopIteration:
            warnings.add("Upper bound not in D_list; defaulting k_max=len(D_list)-1.")
            k_max = len(D_list) - 1

        # Clip to valid range
        k_min = max(0, min(k_min, len(D_list) - 1))
        k_max = max(0, min(k_max, len(D_list) - 1))

        if k_min > k_max:
            warnings.add("Resolved k_min > k_max; resetting to full range.")
            k_min, k_max = 0, len(D_list) - 1

        # Redefine K as {k_min, ..., k_max}
        K = range(k_min, k_max + 1)

        # E_D[K[0] - 1] = E_D[0] and objective base D_list[K[0]-1]
        # (Note: if K[0]==0, this uses E_D[-1] and D_list[-1] by Python semantics.)
        if len(D_list) > 0 and len(list(K)) > 0:
            E_D[list(K)[0] - 1] = E_D[0]

        # ---- Solve the relaxed NF to get fractional x,z for separation ----
        try:
            _, _, x_vals, z_vals = self._max_min_compact_relaxed(n, p_median, q, time_limit)
        except Exception as e:
            warnings.add(f"Relaxed NF solve failed: {e}")
            logger.error("Relaxed NF solve failed.", exc_info=True)
            return None, time.time() - start_time, warnings

        # Identify violated inequalities
        violated = NFwithBoundsCuts.find_violated_inequalities(n, K, E, E_D, x_vals, z_vals)

        # ---- Build final compact model with bounds and cuts ----
        m = Model(name="NFC*_P-Dispersion")

        xi = [i for i in range(n)]
        x = m.binary_var_dict(keys=xi, name="x")
        zk = [k for k in K]
        z = m.binary_var_dict(keys=zk, name="z")

        [m.add_constraint(m.sum([x[i] for i in range(n)]) == p_median)]
        for k in list(K)[1:]:
            m.add_constraint(z[k] <= z[k - 1])

        for k in K:
            for (i, j) in E_D[k] - E_D[k - 1]:
                m.add_constraint(x[i] + x[j] + z[k] <= 2)

        for (k, S) in violated:
            m.add_constraint(m.sum(x[i] for i in S) + (len(S) - 1) * z[k] <= len(S))

        m.maximize(D_list[list(K)[0] - 1] + sum((D_list[k] - D_list[k - 1]) * z[k] for k in K))
        m.parameters.timelimit = time_limit

        try:
            solution = m.solve()
        except Exception as e:
            msg = f"NFC* optimisation failed with exception: {str(e)}"
            warnings.add(msg)
            logger.error(msg, exc_info=True)
            return None, time.time() - start_time, warnings

        runtime = time.time() - start_time

        if solution is None:
            msg = "NFC* model returned no solution (infeasible or solver error)."
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


    def optimise(
        self,
        problem: ProblemData,
        p_median: int,
        validation: bool = False
    ) -> Tuple[Set[str], pd.DataFrame]:
        """
        The main optimise function of the NFC* method for p-dispersion problems.

        Returns:
            (warnings, results_df) where results_df has:
            - algorithm
            - p_median
            - optimal_distance
            - runtime_seconds
            - time_limit
            - status
        """
        optimisation_warnings: Set[str] = set()


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

        logger.info(f"Starting NFC* optimisation for {n} facilities, p={p_median}")

        try:
            opt_value, runtime, nf_warnings = self.nf_with_bounds_cuts(
                n=n,
                p_median=p_median,
                q=q,
                time_limit=self.time_limit,
            )
            optimisation_warnings.update(nf_warnings)

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
                    f"NFC* completed: distance={opt_value:.6f}, runtime_seconds={runtime:.3f}s"
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
                msg = f"NFC* failed to find solution for p={p_median}"
                optimisation_warnings.add(msg)
                logger.error(msg)

        except Exception as e:
            err = f"NFC* optimisation failed with exception: {str(e)}"
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


