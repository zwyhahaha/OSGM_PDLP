# OSGM Adaptive Stepsize Design

## Summary

Replace the constant-stepsize inner PDHG in `osgm_pdhg.jl` with an adaptive (backtracking) stepsize, fix the M-norm to use a stable reference stepsize, and simplify the null-step rule by absorbing it into the existing restart scheme.

## Motivation

The current OSGM implementation uses a constant stepsize (from the power method) for inner PDHG steps. This is overly conservative. The adaptive stepsize used in plain PDHG grows when iterations are easy and shrinks when they are not, leading to faster convergence. Additionally, the M-norm null-step rule (comparing `phi_trial` vs `phi_last`) is a weak proxy for solution quality; the restart scheme's KKT-residual comparison is strictly better and already available.

## Design Decisions

### 1. Reference stepsize for M-norm (`ref_step_size`)

The M-norm `‖r‖_M` with `M = diag(1/τ_p, 1/τ_d)` requires a fixed `τ` to be meaningful across blocks. We use the power-method result `ref_step_size = (1 - ε) / σ_max(A)` (already computed at initialization) as the permanent M-norm reference. This is independent of the adaptive `solver_state.step_size` which evolves during inner steps.

`ref_step_size` is stored in `OsgmPdhgParameters`. `solver_state.step_size` is initialized to `ref_step_size` but drifts freely during adaptive inner steps.

### 2. Adaptive inner stepsize

Each inner PDHG step uses the same interaction/movement backtracking loop as `take_step!(AdaptiveStepsizeParams, ...)` in `primal_dual_hybrid_gradient_gpu.jl`, but calling `compute_next_primal_osgm!` / `compute_next_dual_osgm!` instead of the plain kernels. `solver_state.step_size` is updated after each inner step.

### 3. Block size tied to restart frequency

`osgm_block_size` is removed from `OsgmPdhgParameters`. The number of inner steps per outer iteration `m` is read from `params.restart_params.restart_frequency_if_fixed`. This makes the outer OSGM iteration coincide exactly with the restart evaluation point.

### 4. Null-step replaced by restart scheme

The old null-step (compare M-norm `phi_trial` vs `phi_last`) is removed entirely. Instead:

1. After `m` inner steps, compute `z_cand = z^k - P_k ⊙ R^k` and write it into `current_primal/dual_solution`.
2. Call `run_restart_scheme` — it computes KKT for `z_cand` (now "current") and the weighted average, picks the better point, resets the accumulator if restarting, and updates primal weight.
3. The accepted point (z_cand or weighted average) becomes the new iterate.

No changes to `run_restart_scheme`. No extra KKT cost relative to the existing restart scheme.

### 5. Probe step uses `ref_step_size`

`run_probe_step!` computes the single-step residual `r^k = z^k - T(z^k)` needed for the hypergradient. It uses `ref_step_size` (not `solver_state.step_size`) so the M-norm is stable.

### 6. Hypergradient uses `ref_step_size`

`update_osgm_preconditioner!` derives `primal_step_size = ref_step_size / primal_weight` and `dual_step_size = ref_step_size * primal_weight`. These replace the current lines that use `solver_state.step_size`. `phi_cur` is computed by probing at `z^k` before the inner steps; `phi_next` is computed by probing at the accepted point after the restart scheme.

## Changes

| Location | Change |
|---|---|
| `OsgmPdhgParameters` | Remove `osgm_block_size`; add `ref_step_size::Float64` |
| `CuOsgmState` | Remove `phi_last::Float64` |
| `initialize_osgm_state` | Remove `phi_last` initialization |
| `optimize` (OSGM) | Keep power method; store result as `ref_step_size` in params; derive `m` from `params.restart_params.restart_frequency_if_fixed`; init `solver_state.step_size = ref_step_size` |
| `take_osgm_inner_step!` | Add interaction/movement backtracking loop; update `solver_state.step_size` per inner step |
| `run_probe_step!` | Accept `ref_step_size::Float64` argument; use it instead of `solver_state.step_size` |
| `update_osgm_preconditioner!` | Accept `ref_step_size::Float64` argument; use it for `primal_step_size` / `dual_step_size` |
| `osgm_boundary_step!` | Remove phi null-step; probe at z^k for `phi_cur`; compute z_cand → set current → call `run_restart_scheme` → probe at accepted point for `phi_next` → update hypergradient |
| `take_step!` (OSGM outer) | Pass `ref_step_size` through to `run_probe_step!` and `update_osgm_preconditioner!` |

## Removed

- `phi_last` field and all references
- `osgm_block_size` field and all references (including CLI flag `--osgm_block_size`)
- M-norm null-step comparison (`phi_trial <= osgm_state.phi_last`)
- Power-method initialization of `solver_state.step_size` as constant — now it is an adaptive starting point

## Sanity Check

The existing sanity check (`osgm_stepsize = 0.0`) must still pass: with `osgm_stepsize = 0.0` and preconditioner = I, `z_cand = T^m(z^k)`. The restart scheme then picks between `T^m(z^k)` and the weighted average — the same choice plain PDHG makes at each restart. This is not identical to plain PDHG anymore (plain PDHG has restart frequency driven by KKT adaptively), so the sanity check should be updated to compare against a fixed-frequency PDHG run with the same `restart_frequency_if_fixed`.
