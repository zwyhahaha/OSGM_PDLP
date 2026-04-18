# OSGM-PDHG Implementation Design

**Date:** 2026-04-17  
**Reference implementation:** `asset/residual_feedback_lp.py` (`pdhg_osgm_block`)  
**Template:** `src/adaptive_pdhg.jl`

---

## Overview

Add OSGM (Online Stochastic Gradient Method) preconditioner updates to the existing cuPDLP PDHG loop. The OSGM preconditioner is a diagonal matrix `P_k = diag(primal_hyperparam, dual_hyperparam)` that multiplies into the block residual to compute a candidate iterate `z_cand`. A null-step check gates acceptance; the preconditioner is updated via a hypergradient of the M-norm residual.

The existing restart scheme, adaptive primal weight, and termination logic are **unchanged**. OSGM adds one outer-loop operation at every `osgm_block_size` inner PDHG steps.

---

## Architecture

### New File: `src/osgm_pdhg.jl`

Contains all OSGM-specific structs and the `optimize(params, osgm_params, problem)` dispatch. Included in `src/cuPDLP.jl` after `adaptive_pdhg.jl`.

Inner PDHG steps reuse `take_step!` from `adaptive_pdhg.jl`, passing `primal_hyperparam`/`dual_hyperparam` from `CuOsgmState` as the diagonal preconditioner.

---

## Structs

### `OsgmPdhgParameters` (immutable, CPU)

```julia
struct OsgmPdhgParameters
    osgm_stepsize::Float64   # η: learning rate for preconditioner OGD update
    osgm_block_size::Int64   # m: inner PDHG steps per outer OSGM iteration
end
```

`osgm_block_size` should equal `termination_evaluation_frequency` so the OSGM update coincides with the restart check.

### `CuOsgmState` (mutable, GPU)

```julia
mutable struct CuOsgmState
    z_start_primal::CuVector{Float64}         # z^k primal at block start
    z_start_dual::CuVector{Float64}           # z^k dual at block start
    block_endpoint_primal::CuVector{Float64}  # T^m(z^k) primal, saved before null-step
    block_endpoint_dual::CuVector{Float64}    # T^m(z^k) dual, saved before null-step
    primal_hyperparam::CuVector{Float64}      # diagonal preconditioner, init ones
    dual_hyperparam::CuVector{Float64}        # diagonal preconditioner, init ones
    phi_last::Float64                         # M-norm φ_cur (carried from previous block)
end
```

Initialized in `optimize`: `primal/dual_hyperparam = CUDA.ones(...)`, `phi_last = Inf` (so first block always accepts z_cand).

### `CuProbeBuffer` (mutable, GPU)

Scratch space for the probe PDHG step. Its fields feed both the M-norm computation and the hypergradient.

```julia
mutable struct CuProbeBuffer
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    delta_primal_product::CuVector{Float64}   # A * delta_primal  (free from probe step)
    delta_dual_product::CuVector{Float64}     # A^T * delta_dual  (free from probe step)
    hyper_tmp::CuVector{Float64}              # A^T * dmrlam  (size: num_variables)
    hyper_tmp2::CuVector{Float64}             # A * hyper_tmp (size: num_constraints)
end
```

---

## Outer Loop: OSGM Boundary Operations

Runs every `osgm_block_size` inner steps, immediately before the existing restart/termination check.

```
1.  SAVE z_start
    osgm_state.z_start_primal .= solver_state.current_primal_solution
    osgm_state.z_start_dual   .= solver_state.current_dual_solution

2.  RUN osgm_block_size inner PDHG steps via take_step!
    (preconditioned by osgm_state.primal/dual_hyperparam)
    Weighted average accumulates normally.

3.  SAVE block endpoint T^m(z^k) before null-step overwrites current state:
    osgm_state.block_endpoint_primal .= solver_state.current_primal_solution
    osgm_state.block_endpoint_dual   .= solver_state.current_dual_solution

4.  COMPUTE block residual (GPU kernel):
    R_primal = z_start_primal - current_primal_solution
    R_dual   = z_start_dual   - current_dual_solution

5.  COMPUTE z_cand (GPU kernel):
    z_cand_primal = z_start_primal - primal_hyperparam ⊙ R_primal
    z_cand_dual   = z_start_dual   - dual_hyperparam   ⊙ R_dual

6.  PROBE at z_cand:
    a. Set current iterate to z_cand
    b. Recompute current_primal_product = A * z_cand_primal  (cuSPARSE mv!)
       Recompute current_dual_product   = A^T * z_cand_dual  (cuSPARSE mv!)
    c. Run one PDHG step → writes into probe_buffer
       (delta_primal, delta_dual, delta_primal_product, delta_dual_product all populated)
    d. phi_trial = M_norm(probe_buffer, tau, primal_weight)
       M_norm^2 = ||delta_primal||^2 / (tau * primal_weight)
                + 2 * delta_dual^T * delta_primal_product
                + ||delta_dual||^2 * primal_weight / tau

7.  NULL-STEP DECISION:
    if phi_trial ≤ osgm_state.phi_last:
        ACCEPT: current iterate stays at z_cand
        phi_next = phi_trial
        probe_buffer holds r_next data
    else:
        REJECT: restore T^m(z^k) from block_endpoint_*
                recompute current_primal/dual_product via cuSPARSE mv!
        Run one PROBE at T^m(z^k) → overwrites probe_buffer
        phi_next = M_norm(probe_buffer, tau, primal_weight)

8.  RUN existing restart scheme (run_restart_scheme):
    Compares current iterate vs weighted average on KKT residual.
    Picks better as new restart point.

9.  UPDATE primal weight if restart occurred (compute_new_primal_weight).

10. COMPUTE OSGM hypergradient (skip if phi_last * phi_next < 1e-30 or osgm_stepsize == 0):
    Inputs: probe_buffer at z^{k+1}, R_primal, R_dual, osgm_state.phi_last, phi_next
    active_dual[j] = (current_dual_solution[j] > 0)  at z^{k+1}

    a. mx   = delta_primal / tau + delta_dual_product   [free: delta_dual_product = A^T delta_dual]
    b. mlam = delta_primal_product + delta_dual / tau   [free: delta_primal_product = A delta_primal]
    c. dmrlam = active_dual ⊙ mlam
    d. hyper_tmp  = A^T * dmrlam                        [SpMV on constraint_matrix_t]
    e. hyper_tmp2 = A  * hyper_tmp                      [SpMV on constraint_matrix]
    f. g_primal = sigma * hyper_tmp ⊙ R_primal
       g_dual   = (-tau * (A * mx) + (1 - active_dual) ⊙ mlam
                   + 2*sigma*tau * hyper_tmp2) ⊙ R_dual
       [Note: A * mx requires one more SpMV on constraint_matrix]
    g. OGD update (plain gradient, no momentum):
       primal_hyperparam .= max.(0, primal_hyperparam .+ osgm_stepsize .* g_primal ./ (phi_last * phi_next))
       dual_hyperparam   .= max.(0, dual_hyperparam   .+ osgm_stepsize .* g_dual   ./ (phi_last * phi_next))

11. CARRY FORWARD:
    osgm_state.phi_last = phi_next
```

---

## M-norm Formula

Using the full operator metric (not diagonal approximation). Let:
- `τ_p = step_size / primal_weight`  (primal step size)
- `τ_d = step_size * primal_weight`  (dual step size)

```
‖(r_x, r_lam)‖²_M = r_x^T (r_x/τ_p + A^T r_lam) + r_lam^T (A r_x + r_lam/τ_d)
                   = ‖r_x‖²/τ_p + 2 r_lam^T (A r_x) + ‖r_lam‖²/τ_d
```

In code: `delta_primal_product = A * delta_primal` and `delta_dual_product = A^T * delta_dual` are both free from the probe step, so M-norm costs only two dot products and two norm-squareds — no extra SpMV.

In the hypergradient (step 10), `tau = τ_p` and `sigma = τ_d` following the Python convention.

---

## Hypergradient SpMV Count

Per outer step:
- `A^T * delta_dual` — free (delta_dual_product from probe)
- `A * delta_primal` — free (delta_primal_product from probe)
- `A^T * dmrlam` — 1 new SpMV
- `A * hyper_tmp` — 1 new SpMV
- `A * mx` — 1 new SpMV (for g_dual vlam term)

Total new SpMV: 3 (not 2 as earlier estimated — `A * mx` is also needed).

---

## Sanity Check Invariant

With `osgm_stepsize = 0.0` and `primal/dual_hyperparam` initialized to ones:
- `z_cand = z_start - 1 ⊙ R = T^m(z^k)` → null-step always accepts (phi_trial = M-norm at T^m)
- Preconditioner update is skipped (`osgm_stepsize == 0`)
- Inner PDHG steps are unscaled (hyperparam = ones passes through unchanged)
- Must produce **identical iterate sequence** to plain PDHG

Verified via `test/osgm_sanity_check.jl` on `data/netlib/afiro.mps.gz` (should terminate < 60s).

---

## Entry Point

```julia
function optimize(
    params::PdhgParameters,
    osgm_params::OsgmPdhgParameters,
    original_problem::QuadraticProgrammingProblem,
)
```

Dispatches alongside the existing `optimize(params, problem)` in `adaptive_pdhg.jl`. Wired from `scripts/solve.jl` via `--osgm_stepsize` and `--osgm_block_size` flags (already present).

---

## Files Changed

| File | Change |
|---|---|
| `src/osgm_pdhg.jl` | New file: `OsgmPdhgParameters`, `CuOsgmState`, `CuProbeBuffer`, `optimize` dispatch |
| `src/cuPDLP.jl` | Add `include("osgm_pdhg.jl")` after `adaptive_pdhg.jl` |
| `test/osgm_sanity_check.jl` | Sanity check (already exists, verify it exercises the new dispatch) |
