---
name: cupdhg:algo
description: Use when implementing a new algorithmic variant in cuPDLP.jl, modifying restart logic, step size policy, or preconditioning, or when understanding the math behind PDHG iteration, OSGM hypergradient update, M-norm, block residual, or primal weight update. Also use before reading osgm_pdhg.jl or primal_dual_hybrid_gradient_gpu.jl to understand what the code does mathematically.
---

# cuPDLP Algorithm Expert

See `references/formulas.md` for dense math notation.

## Problem Form

Standard form LP/QP: `min ½xᵀQx + cᵀx` s.t. `Ax = b` (equalities), `Ax ≤ b` (inequalities), `l ≤ x ≤ u`.

The saddle-point form is `min_x max_y L(x,y) = cᵀx + yᵀ(Ax-b) - δ(x)` where `δ` encodes bounds.

## PDHG Iteration (one step)

Variables: `x` (primal, size `num_variables`), `y` (dual, size `num_constraints`).
Parameters: `τ` = `step_size`, `ω` = `primal_weight`, `α` = `extrapolation_coefficient` (= `pock_chambolle_alpha`, default 1.0).

**Primal update** (`compute_next_primal_solution_kernel!`, line 177 of `primal_dual_hybrid_gradient_gpu.jl`):
```
Δx = Π_{[l,u]}(x - (τ/ω)(c - Aᵀy)) - x
```
Box projection: `clamp(·, l, u)`. The `Aᵀy` product (`current_dual_product`) is pre-computed via cuSPARSE.

**Dual update** (`compute_next_dual_solution_kernel!`, line 232):
```
Δy = Π_{y≥0 for ineq}(y + (ωτ)(b - A(x + (1+α)Δx + α·Δx_prev))) - y
```
Equality rows: no projection. Inequality rows: `max(·, 0)`. The `(1+α)·AΔx` extrapolation is Pock-Chambolle; with `α=1` this is the standard 2-step extrapolation.

**State advance** (`update_solution_in_solver_state!`, line 297):
- `x ← x + Δx`, `y ← y + Δy`
- `Aᵀy` updated via cuSPARSE `mv!` on new `y`
- Weighted average accumulates with weight = `step_size`

## Step Size Policies

### AdaptiveStepsizeParams (default)
No power method. Initial step = `1 / ‖A‖_∞` (max row sum).

Each step: compute `interaction = |Δxᵀ · AᵀΔy|` and `movement = ‖Δx‖²/τ·ω + ‖Δy‖²·τ·ω`.
If `interaction > movement`: backtrack — multiply `step_size` by `(0.5)^reduction_exponent`.
Otherwise: grow — multiply by `(1 + iter^(-growth_exponent))`.

`ratio_step_sizes` and `required_ratio` track this in `CuPdhgSolverState`.

### ConstantStepsizeParams
Calls `estimate_maximum_singular_value` (power method, ~20 iterations) at startup.
Sets `step_size = (1 - 0.2) / σ_max`. Power iterations count toward `cumulative_kkt_passes`.
**Recommended for OSGM** because OSGM assumes a fixed step throughout.

## Restart Scheme

Evaluated every `termination_evaluation_frequency` iterations (first 10: every iteration).

### ADAPTIVE_KKT (default)
Computes KKT residual ρ for both current iterate and weighted average:
```
ρ = sqrt(ω·‖primal_res‖² + (1/ω)·‖dual_res‖² + |gap|²)
```
Restart triggers if either:
- `ρ_candidate / ρ_last_restart < sufficient_reduction` (default 0.2): unconditional restart
- Ratio in `[sufficient, necessary]` (default 0.8) AND worse than last trial: restart

### KKT_GREEDY (RestartToCurrentMetric)
Picks whichever of {current iterate, weighted average} has smaller ρ as the new start point.
Otherwise always restarts to weighted average.

### Artificial restart
If no restart occurred in `artificial_restart_threshold` (default 0.36) × total_iterations: force restart.

## Primal Weight Update (on every restart)

```
ω_new = exp(s · log(‖Δdual‖/‖Δprimal‖) + (1-s) · log(ω_old))
```
where `s` = `primal_weight_update_smoothing` (default 0.5), and distances are measured since the previous restart point.

Implemented in `compute_new_primal_weight` (`saddle_point_gpu.jl:627`).

## Preconditioning (Ruiz Equilibration)

`preprocess.jl#ruiz_rescaling`: alternating L∞ row/column normalization of `A`, `l_inf_ruiz_iterations` rounds (default 10). Optional L2 pass (`l2_norm_rescaling`).

Produces diagonal `variable_rescaling` and `constraint_rescaling`; stored in `ScaledQpProblem`. Solutions are unscaled in `saddle_point_gpu.jl#unscaled_saddle_point_output` before returning.

## OSGM-PDHG Outer Loop (`osgm_pdhg.jl#optimize`, line 686)

Each outer step covers `block_size` (= m) inner PDHG steps:

1. **Single-step residual**: `r^k = z^k - T(z^k)` where `T` is one PDHG step. Compute `φ_cur = ‖r^k‖_M` (M-norm, see `references/formulas.md`).

2. **Block residual**: run `m-1` more PDHG steps to get `T^m(z^k)`; form `R^k = z^k - T^m(z^k)`.

3. **OSGM candidate**: `z_cand = z^k - P_k ⊙ R^k` where `P_k = diag(prec_primal, prec_dual)`.

4. **Null-step**: if `φ(z_cand) ≤ φ_cur` → accept `z_cand`; else fall back to `T^m(z^k)`.

5. **Preconditioner update** (skipped when `osgm_stepsize = 0`):
   - Compute `φ_next = ‖r^{k+1}‖_M`
   - Hypergradient: `g^k = (I - J_T(z^{k+1}))ᵀ M r^{k+1} ⊙ R^k / (φ_cur · φ_next)`
     computed by `update_hyper_state!` (line 619) via `update_primal_hyper_state!` + `update_dual_hyper_state!`
   - Update: `p ← max(0, p + η · g^k)` where η = `osgm_stepsize`
   - Guard: skip update if `φ_cur · φ_next < 1e-30`

6. **Primal weight + weighted average**: same exponential-smoothing formula as restart scheme; applied every outer step.

**Sanity check**: `osgm_stepsize = 0.0` keeps `P_k = I`, so `z_cand = T^m(z^k)` always accepted → identical to plain PDHG. Verified in `test/osgm_sanity_check.jl`.

## How to Add a New Algorithmic Variant

1. **Define parameters struct** (alongside `OsgmPdhgParameters` in `osgm_pdhg.jl`):
   ```julia
   struct MyVariantParams
       my_param::Float64
   end
   ```

2. **Define state struct** if GPU state is needed (alongside `CuOsgmState` pattern):
   ```julia
   mutable struct CuMyVariantState
       my_gpu_vec::CuVector{Float64}
   end
   ```
   Initialize with `CUDA.ones(Float64, n)` or `CUDA.zeros(Float64, n)` in the `optimize` entry point.

3. **Implement the inner loop** as a new `optimize` dispatch:
   ```julia
   function optimize(
       params::PdhgParameters,
       my_params::MyVariantParams,
       problem::QuadraticProgrammingProblem,
   )
       # setup: scale, transfer to GPU, init state
       # call existing take_step! or pdhg_step_pure! for inner PDHG
       # apply your variant logic
       # call evaluate_unscaled_iteration_stats + check_termination_criteria
   end
   ```

4. **Wire termination/logging**: reuse `compute_iteration_stats` (`iteration_stats_utils_gpu.jl:377`) and `check_termination_criteria` (`termination.jl`). Do not reimplement KKT checks.

5. **Add sanity check test** at `test/` verifying `my_param=0` (or equivalent) recovers plain PDHG behavior, following `test/osgm_sanity_check.jl` pattern.
