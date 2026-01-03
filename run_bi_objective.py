
from pathlib import Path
import pandas as pd
from src.data_class import ProblemData
from src.maxmin_diversity.bisection import BisectionMethod
from src.bi_objective.epsilon_method import BiObjectiveEpsilonMethod


# --- Runner ---
def run_bi_obj():
    all_results = []  # Collect everything in one list

    for n in N_VALUES:
        data = pd.read_csv(f"data/paper_data/GKD_data_{n}.txt", sep=r"\s+", header=None)

        problem_data = ProblemData(data=data)
        p_values = [round(r * n) for r in P_RATIOS]

        for p_median in p_values:
            # Step 1: Bisection to get min_distance
            solver = BisectionMethod(ratio=0.3)
            _, bisect_df = solver.optimise(problem=problem_data, p_median=p_median)

            if "optimal_distance" not in bisect_df.columns:
                raise KeyError(f"'optimal_distance' column not found. Columns: {list(bisect_df.columns)}")

            min_distance = bisect_df["optimal_distance"].iloc[0]
            print(f"[n={n}, p={p_median}] min_distance = {min_distance}")

            # Step 2: Optimizer
            optimizer = BiObjectiveEpsilonMethod(
                problem_data=problem_data,
                p_median=p_median,
                min_distance=min_distance
            )

            # Step 3: Run all methods and collect results
            for method in METHODS:
                results_dict = optimizer.epsilon_method(
                    method=method,
                    epsilon=EPSILON,
                    cutting_plane=True
                )

                # Flatten results_dict into rows
                for sol_id, solution in results_dict.items():
                    all_results.append({
                        "n": n,
                        "p": p_median,
                        "method": method,
                        "solution_id": sol_id,
                        "objective_value": solution.get("objective_value"),
                        "achieved_beta": solution.get("achieved_beta"),
                        "iter_runtime": solution.get("iter_runtime"),
                        "iteration": solution.get("iteration", sol_id),
                        "beta_target": solution.get("beta_target", 0),
                        "num_selected_facilities": len(solution.get("selected_facilities", [])),
                        "selected_facilities": str(solution.get("selected_facilities", [])),
                        "num_pareto_solutions": solution.get("num_pareto_solutions"),
                        "total_runtime": solution.get("total_runtime"),
                    })

    # Step 4: Save everything in one CSV
    output_file = Path.cwd() / "all_pareto_results.csv"
    pd.DataFrame(all_results).to_csv(output_file, index=False)
    print(f" All results saved to: {output_file}")

# --- Config ---
if __name__ == "__main__":
    N_VALUES = [50,100,150,200,250,300]
    P_RATIOS = [0.1, 0.2, 0.3]
    EPSILON = 1
    METHODS = ["quadratic", "cutting_plane", "benders", "branch_cut_benders"]

    run_bi_obj()
