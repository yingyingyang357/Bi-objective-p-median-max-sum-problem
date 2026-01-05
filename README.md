# Bi-Objective p-Median and Max-Sum Diversity Problem
This repository contains algorithms and implementations to solve:

- Max-Min Diversity Problem (MMDP)
- Bi-Objective p-Median and Max-Sum Diversity Problem (Bp-MMSDP)

# Algorithms Implemented

**For the Max-Min Diversity Problem**

- Improved Bisection Search (IBSS)[1]
- Procedure A (ProcA) [2]
- Direct Binary Search (DBC) [3]
- NF, NF*, NFC* [4]

**For the Bi-Objective p-Median and Max-Sum Diversity Problem [1]**

Epsilon-Constraint Method incorporating:
- Quadratic Programming (εC)
- Tangent Cutting Plane (εC+TCP)
- Benders Decomposition with Balinski Scheme and Tangent Cutting Plane (εC+TCP+BD)
- Branch-and-Cut Version of Benders Decomposition (εC+TCP+BC)

Note that the TCP method can be disabled in Benders decomposition if needed.


# Repository Structure

```
BI-OBJECTIVE-P-MEDIAN-MAX-SUM-PROBLEM/
├── data/
│   ├── GKD-d data/        # Benchmark datasets
│   ├── paper_data/        # Datasets for experiments
│   └── CAB data.txt       # Benchmark datasets
├── notebooks/             # Jupyter notebooks for tests and visulizations
├── src/
│   ├── bi_objective/
│   │   └── epsilon_method.py    # Main Epsilon-constraint implementation
│   ├── maxmin_diversity/    # MMDP methods
│   │   ├── binary_search.py   # (ProcA)
│   │   ├── bisection.py       # (IBSS)
│   │   ├── direct_search.py   # (DBC)
│   │   ├── nf_star.py         # (NF*)
│   │   ├── nf.py              # (NF)
│   │   └── nfc_star.py        # (NFC*)
│   ├── p_median/    # P(beta) solver (quadratic model, TCP, TCP+BD, TCP+BC)
│   │   ├── bender_decomposition.py
│   │   ├── benders_master.py
│   │   ├── beta_model.py
│   │   └── p_median_solver.py
│   ├── data_class.py       # Data structure for problem instances
│   └── solver_class.py     # Common solver utilities
├── run_bi_objective.py     # Script to run bi-objective experiments
├── run_maxmin.py           # Script to run max-min diversity experiments
├── requirements.txt        # Dependencies
├── LICENSE
└── README.md
```

# Installation

- Clone the repository:

`git clone https://github.com/your-username/BI-OBJECTIVE-P-MEDIAN-MAX-SUM-PROBLEM.git`

`cd BI-OBJECTIVE-P-MEDIAN-MAX-SUM-PROBLEM`

- Install dependencies:

`pip install -r requirements.txt`


# Usage
- Run Max-Min Diversity Problem

  `python run_maxmin.py`

- Run Bi-Objective p-Median & Max-Sum Diversity

  `python run_bi_objective.py`

You can configure:

- n (number of locations)
- p (number of open facilities)
- epsilon (epsilon value)
- methods (algorithm chosen)


# Dependencies

- Cplex 22.1.1

# References

[1] Yingying Yang, Hoa T. Bui, Ryan Loxton. An Exact Method for the Bi-objective p-median Max-sum Diversity Problem.

[2] Sayyady, F., Fathi, Y., 2016. An integer programming approach for solving the p-dispersion problem. European Journal of Operational Research 253, 216–225. doi:10.1016/j.ejor.2016.02.026.

[3] Parre˜no, F., ´Alvarez-Vald´es, R., Mart´ı, R., 2021. Measuring diversity. A review and an empirical analysis. European Journal of Operational Research 289, 515–532. doi:10.1016/j.ejor.2020.07.053.

[4] Sayah, D., Irnich, S., 2017. A new compact formulation for the discrete p-dispersion problem. European Journal of Operational Research 256, 62–67. doi:10.1016/j.ejor.2016.06.036.