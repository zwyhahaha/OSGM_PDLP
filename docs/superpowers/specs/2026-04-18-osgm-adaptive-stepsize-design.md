# OSGM Adaptive Stepsize Design

## Summary

Replace the fixed-block-size outer OSGM loop with an event-driven design: run adaptive PDHG steps continuously, and trigger the OSGM boundary update (z_cand, hypergradient) only when the adaptive restart scheme fires. Fix the M-norm to use a stable reference stepsize.

## Motivation

The current OSGM implementation uses a fixed block size `m` and a constant stepsize for inner PDHG steps. Both are suboptimal: adaptive stepsize grows when iterations are easy and shrinks when they are not; tying OSGM updates to adaptive restarts means the block size is determined by actual convergence progress rather than a hand-tuned parameter. The M-norm null-step rule (comparing `phi_trial` vs `phi_last`) is a weak proxy; the restart scheme's KKT comparison is strictly better and already available at zero extra cost.

## Design

### Main loop structure

The outer OSGM loop (fixed `m` steps per outer iteration) is replaced by the plain adaptive PDHG step loop — one step per iteration, identical in structure to `primal_dual_hybrid_gradient_gpu.jl`. The OSGM boundary update fires only when the adaptive restart scheme decides to restart.

### Adaptive inner stepsize

Each PDHG step uses the interaction/movement backtracking loop from `take_step!(AdaptiveStepsizeParams, ...)`, but calling `compute_next_primal_osgm!` / `compute_next_dual_osgm!` instead of the plain kernels. `solver_state.step_size` updates after each step.

### Reference stepsize for M-norm (`ref_step_size`)

The M-norm `‖r‖_M` requires a fixed `τ` to be stable across restarts. We use the power-method result `ref_step_size = (1 - ε) / σ_max(A)` computed at initialization as the permanent M-norm reference. `solver_state.step_size` is initialized to `ref_step_size` but drifts freely during adaptive steps. `ref_step_size` is stored in `OsgmPdhgParameters`.

### OSGM update on restart

`z_start_primal/dual` in `CuOsgmState` records the iterate at the last restart point. When the adaptive restart scheme fires:

1. Compute block residual `R^k = z_start - current`
2. Compute `z_cand = z_start - P_k ⊙ R^k`; write into `current_primal/dual_solution`; `recompute_a_products!`
3. Call `run_restart_scheme` — computes KKT for z_cand ("current") and weighted average, picks the better point, resets accumulator, updates primal weight. No changes to `run_restart_scheme`.
4. Probe at accepted point with `ref_step_size` → `phi_next`
5. Update hypergradient using `phi_last × phi_next`
6. Save `phi_next → phi_last`; save accepted point → `z_start`

If the restart scheme does not restart, no OSGM update occurs and `z_start` is not changed.

### `phi_last` role

`phi_last` is kept in `CuOsgmState` but its meaning changes: it is the M-norm at the last restart point (not the last block start). It is initialized to `Inf` so the hypergradient guard (`phi_prod < 1e-30`) suppresses the first update until two restart points exist.

### `run_probe_step!` and `update_osgm_preconditioner!`

Both use `ref_step_size` instead of `solver_state.step_size`, so the M-norm and Jacobian approximation remain stable regardless of adaptive step drift.

## Changes

| Location | Change |
|---|---|
| `OsgmPdhgParameters` | Remove `osgm_block_size`; add `ref_step_size::Float64` |
| `CuOsgmState` | Remove `block_endpoint_primal/dual`; keep `z_start_primal/dual`, `phi_last`, `primal_hyperparam`, `dual_hyperparam` |
| `initialize_osgm_state` | Remove `block_endpoint` initialization |
| `optimize` (OSGM) | Keep power method; store result as `ref_step_size`; replace outer OSGM loop with plain per-step loop; on restart trigger, call OSGM boundary update |
| `take_osgm_inner_step!` | Add interaction/movement backtracking loop; update `solver_state.step_size` per step |
| `run_probe_step!` | Accept `ref_step_size::Float64`; use it instead of `solver_state.step_size` |
| `update_osgm_preconditioner!` | Accept `ref_step_size::Float64`; use it for `primal_step_size` / `dual_step_size` |
| `osgm_boundary_step!` | Called only on restart; compute z_cand from z_start and current; set current = z_cand; call `run_restart_scheme`; probe phi_next; update hypergradient; save phi_last and z_start |
| `take_step!` (OSGM outer) | Removed — replaced by per-step loop in `optimize` |

## Removed

- `osgm_block_size` field and all references (including CLI flag `--osgm_block_size`)
- `block_endpoint_primal/dual` from `CuOsgmState`
- Fixed outer OSGM loop (`take_step!` for OSGM)
- M-norm null-step comparison (`phi_trial <= osgm_state.phi_last`)

## Sanity Check

With `osgm_stepsize = 0.0` and preconditioner = I, `z_cand = z_start - R^k = current` (since `P_k = I` and `z_cand = z_start - (z_start - current) = current`). The restart scheme then picks between current and weighted average exactly as plain PDHG does. So `osgm_stepsize = 0.0` recovers plain adaptive PDHG with the same restart scheme — the existing sanity check in `test/osgm_sanity_check.jl` should verify this.
