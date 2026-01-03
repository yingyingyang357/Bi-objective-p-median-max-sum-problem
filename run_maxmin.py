
# run_all_methods_save_csv.py
import os
import logging
import pandas as pd
from src.data_class import ProblemData
from src.maxmin_diversity.bisection import BisectionMethod
from src.maxmin_diversity.binary_search import BinarySearch
from src.maxmin_diversity.direct_search import DirectBinarySearch
from src.maxmin_diversity.nf import NewCompactFormulation
from src.maxmin_diversity.nf_star import NFwithBounds
from src.maxmin_diversity.nfc_star import NFwithBoundsCuts


# ---------------------------------------------------------------------
# Configurable algorithm registry
# Each entry: ("algorithm_name", callable_factory)
# callable_factory returns a solver instance that has .optimise(problem, p_median)
# ---------------------------------------------------------------------
ALGORITHMS = [
    ("IBSS", lambda: BisectionMethod(ratio=0.3)),
    ("ProcA", lambda: BinarySearch()),
    ("DBS",lambda:DirectBinarySearch()),
    ("NF", lambda: NewCompactFormulation()),
    ("NF*", lambda: NFwithBounds()),
    ("NFC*", lambda: NFwithBoundsCuts()),
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_problem_data_for_n(n: int) -> ProblemData:
    """
    Load data for given n using your new path:
      data/paper_data/GKD_data_{n}.txt
    Four columns per row with whitespace separator (based on your example).
    """
    path = f"data/paper_data/GKD_data_{n}.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )
    
    return ProblemData(data=data)


def safe_optimise(solver, problem: ProblemData, p_median: int):
    """
    Run solver.optimise(problem, p_median) and extract:
      - algorithm (from results_df 'algorithm' column if present; else solver.__class__.__name__)
      - optimal_distance
      - runtime_seconds
    Return a dict and a success flag.
    """
    try:
        warnings, results_df = solver.optimise(problem=problem, p_median=p_median, validation=False)
        if results_df is None or results_df.empty:
            return {
                "algorithm": getattr(solver, "name", solver.__class__.__name__),
                "optimal_distance": None,
                "runtime_seconds": None,
            }, False

        # Expected columns: algorithm, optimal_distance, runtime_seconds, time_limit, status
        algorithm = None
        if "algorithm" in results_df.columns and pd.notna(results_df["algorithm"].iloc[0]):
            algorithm = str(results_df["algorithm"].iloc[0])
        else:
            algorithm = getattr(solver, "name", solver.__class__.__name__)

        optimal_distance = None
        if "optimal_distance" in results_df.columns:
            optimal_distance = results_df["optimal_distance"].iloc[0]

        runtime_seconds = None
        if "runtime_seconds" in results_df.columns:
            runtime_seconds = results_df["runtime_seconds"].iloc[0]

        return {
            "algorithm": algorithm,
            "optimal_distance": optimal_distance,
            "runtime_seconds": runtime_seconds,
        }, True

    except Exception as e:
        logging.exception(f"Optimisation failed for {solver.__class__.__name__}: {e}")
        return {
            "algorithm": getattr(solver, "name", solver.__class__.__name__),
            "optimal_distance": None,
            "runtime_seconds": None,
        }, False


def run_and_collect(
    n_values,
    p_selectors,
    num_runs: int,
    output_csv_path: str,
):
    """
    For each n in n_values and each ratio in p_selectors:
      - Compute p = round(selector * n)
      - Run each algorithm num_runs times
      - Save rows: [n, p, algorithm, optimal_distance, avg_runtime_seconds]
    """
    records = []

    for n in n_values:
        problem = load_problem_data_for_n(n)

        for selector in p_selectors:
            p_median = round(selector * n)

            for alg_name, factory in ALGORITHMS:
                runtimes = []
                best_distance = None

                for _ in range(num_runs):
                    solver = factory()
                    result, ok = safe_optimise(solver, problem, p_median=p_median)

                    rt = result.get("runtime_seconds", None)
                    if rt is not None:
                        try:
                            runtimes.append(float(rt))
                        except Exception:
                            pass

                    od = result.get("optimal_distance", None)
                    if od is not None:
                        best_distance = od  # take latest non-None

                avg_runtime = round(sum(runtimes) / len(runtimes), 2) if runtimes else None

                records.append({
                    "n": n,
                    "p": p_median,
                    "algorithm": alg_name,
                    "optimal_distance": best_distance,
                    "avg_runtime_seconds": avg_runtime,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved results to: {output_csv_path}")
    return df

# ---------------------------------------------------------------------
# Run instances
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Choose n values and p values
    n_values = [100]              # extend: [100, 250, 500, 1000, 2000]
    p_selectors = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_runs = 3

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(current_dir, "maxmin_methods_result.csv")

    run_and_collect(
        n_values=n_values,
        p_selectors=p_selectors,
        num_runs=num_runs,
        output_csv_path=output_csv_path,
    )

