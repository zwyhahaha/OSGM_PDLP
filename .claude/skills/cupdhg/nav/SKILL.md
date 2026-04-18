---
name: cupdhg:nav
description: Use when navigating the cuPDLP.jl codebase — finding which file or function handles a behavior, looking up struct definitions and field names, tracing data flow from entry point to GPU kernels, or orienting before making any edit. Invoke this before reading source files to save tokens.
---

# cuPDLP.jl Navigation Map

## File Inventory

| File | Purpose |
|---|---|
| `src/cuPDLP.jl` | Module root — imports and `include()` ordering |
| `src/quadratic_programming.jl` | CPU-side QP data: `QuadraticProgrammingProblem` (line 25), `ScaledQpProblem` (line 284) |
| `src/solve_log.jl` | Logging structs: `ConvergenceInformation` (51), `InfeasibilityInformation` (161), `IterationStats` (219), `SolveLog` (336) |
| `src/quadratic_programming_io.jl` | MPS/QPS reader: `qps_reader_to_standard_form` |
| `src/preprocess.jl` | Ruiz scaling: `ruiz_rescaling`, `l2_norm_rescaling`; `ScaledQpProblem` construction; `PresolveInfo` (210) |
| `src/cpu_to_gpu.jl` | CPU→GPU transfer: `CuLinearProgrammingProblem` (1), `CuScaledQpProblem` (16), `load_problem_to_gpu` |
| `src/termination.jl` | `TerminationCriteria` (9), `CachedQuadraticProgramInfo` (123), `construct_termination_criteria`, `check_termination_criteria` |
| `src/iteration_stats_utils_gpu.jl` | KKT residuals on GPU: `compute_iteration_stats`, `evaluate_unscaled_iteration_stats`, `compute_convergence_information` (251) |
| `src/saddle_point_gpu.jl` | Weighted averages, restart logic, primal weight update: `run_restart_scheme` (479), `compute_new_primal_weight` (627), `construct_restart_parameters` (388) |
| `src/primal_dual_hybrid_gradient_gpu.jl` | Core PDHG kernels and plain `optimize()`: `compute_next_primal_solution_kernel!` (177), `compute_next_dual_solution_kernel!` (232), `take_step!` (343/418), `optimize` (457) |
| `src/osgm_pdhg.jl` | OSGM outer loop and `optimize(params, osgm_params, problem)`: `update_hyper_state!` (619), `optimize` (686) |
| `src/MOI_wrapper.jl` | MathOptInterface bridge: `Optimizer` struct, `optimize!` MOI entry point |

## Key Types Index

**CPU-side (standard form):**
- `QuadraticProgrammingProblem` — `quadratic_programming.jl:25` — variables, bounds, objective, constraint matrix (CPU `SparseMatrixCSC`)
- `ScaledQpProblem` — `quadratic_programming.jl:284` — original + scaled QP + `variable_rescaling`, `constraint_rescaling` vectors
- `PresolveInfo` — `preprocess.jl:210` — records singleton constraints removed in presolve

**GPU-side (LP representation):**
- `CuLinearProgrammingProblem` — `cpu_to_gpu.jl:1` — all arrays as `CuVector`/`CuSparseMatrix` (CSR)
- `CuScaledQpProblem` — `cpu_to_gpu.jl:16` — GPU-side scaled problem + rescaling vectors

**PDHG solver state (mutable, GPU):**
- `CuPdhgSolverState` — `primal_dual_hybrid_gradient_gpu.jl:26` (plain PDHG) / `osgm_pdhg.jl:30` (OSGM, adds `learning_rate`, `online_scaling`)
  - Key fields: `current_primal_solution`, `current_dual_solution`, `step_size`, `primal_weight`, `solution_weighted_avg`, `cumulative_kkt_passes`
- `CuBufferState` — `primal_dual_hybrid_gradient_gpu.jl:42` — scratch buffers: `delta_primal`, `delta_dual`, `delta_primal_product`, `delta_dual_product`

**Parameters (immutable, CPU):**
- `PdhgParameters` — `primal_dual_hybrid_gradient_gpu.jl:9` — hyperparameters: `termination_criteria`, `restart_params`, `step_size_policy_params`, `termination_evaluation_frequency`
- `AdaptiveStepsizeParams` — `primal_dual_hybrid_gradient_gpu.jl:2` — `reduction_exponent`, `growth_exponent`
- `ConstantStepsizeParams` — `primal_dual_hybrid_gradient_gpu.jl:7` — singleton, triggers power method
- `PdhgParameters` (OSGM variant) — `osgm_pdhg.jl:9` — extends base params with `learning_rate::Float64`, `online_scaling::Bool`, `online_scaling_frequency::Int64`, `normalize::Bool`
- `TerminationCriteria` — `termination.jl:9` — tolerances, time/iteration limits
- `RestartParameters` — `saddle_point_gpu.jl` — `restart_scheme`, `artificial_restart_threshold`, etc.

**Logging:**
- `ConvergenceInformation` — `solve_log.jl:51` — primal/dual residuals, objectives, gap
- `IterationStats` — `solve_log.jl:219` — per-iteration log entry

## Enums

| Enum | File | Values |
|---|---|---|
| `RestartScheme` | `saddle_point_gpu.jl:343` | `NO_RESTARTS`, `FIXED_FREQUENCY`, `ADAPTIVE_KKT` |
| `RestartToCurrentMetric` | `saddle_point_gpu.jl:350` | `NO_RESTART_TO_CURRENT`, `KKT_GREEDY` |
| `OptimalityNorm` | `termination.jl:6` | `L_INF`, `L2` |
| `TerminationReason` | `solve_log.jl:323` | `OPTIMAL`, `PRIMAL_INFEASIBLE`, `DUAL_INFEASIBLE`, `TIME_LIMIT`, `ITERATION_LIMIT`, `NUMERICAL_ERROR` |
| `PointType` | `solve_log.jl:39` | `CURRENT_ITERATE`, `AVERAGE_ITERATE`, `OUTPUT_ITERATE` |
| `RestartChoice` | `solve_log.jl:19` | `RESTART_CHOICE_UNSPECIFIED`, `RESTART_CHOICE_NO_RESTART`, `RESTART_CHOICE_WEIGHTED_AVERAGE_RESET`, `RESTART_CHOICE_RESTART_TO_AVERAGE` |

## Data Flow

```
scripts/solve.jl  OR  MOI_wrapper.jl#optimize!
    ↓
saddle_point_gpu.jl#optimize()          ← entry point, handles scaling/unscaling
    ↓
preprocess.jl#ruiz_rescaling()          ← Ruiz equilibration (10 rounds default)
    ↓
cpu_to_gpu.jl#load_problem_to_gpu()     ← transfer SparseMatrixCSC → CuSparseMatrix
    ↓
primal_dual_hybrid_gradient_gpu.jl#optimize()   ← plain PDHG loop
  OR osgm_pdhg.jl#optimize()                    ← OSGM outer loop
    ↓ (each iteration)
take_step!()
  → compute_next_primal_solution!()     ← kernel + cuSPARSE mv!
  → compute_next_dual_solution!()       ← kernel
  → update_solution_in_solver_state!()  ← weighted average accumulation
    ↓ (every termination_evaluation_frequency iters)
evaluate_unscaled_iteration_stats()     ← KKT on unscaled average iterate
check_termination_criteria()
```

## State Mutation Map

Functions that mutate `CuPdhgSolverState` in-place:
- `update_solution_in_solver_state!` — advances primal/dual, accumulates weighted avg
- `take_step!` (adaptive) — also updates `step_size`, `required_ratio`
- `run_restart_scheme` — may reset `current_primal_solution`, `current_dual_solution`, `primal_weight`

Functions that return new values (no mutation):
- `compute_next_primal_solution!` / `compute_next_dual_solution!` — write into `CuBufferState`
- `compute_iteration_stats` — pure read, returns `IterationStats`
- `compute_new_primal_weight` — returns new weight scalar
