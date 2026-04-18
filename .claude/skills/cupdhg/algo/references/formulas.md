# Dense Math Formulas

## M-norm

The metric `M = diag(ω/τ, 1/(ωτ))` acts block-diagonally on `z = [x; y]`:

```
‖[r_x; r_y]‖²_M = (ω/τ)‖r_x‖² + (1/(ωτ))‖r_y‖²
```

In code (`osgm_pdhg.jl#compute_norms`): implemented as weighted dot products using `CUDA.dot`.

The off-diagonal `[[0, Aᵀ],[A, 0]]` term in the full M (from the saddle-point structure) is incorporated via the cuSPARSE products when evaluating `‖r‖_M` on the full residual.

## Hypergradient g^k

The OSGM preconditioner update uses the hypergradient of `φ(z_cand)` w.r.t. `p`:

```
g^k = (I - J_T(z^{k+1}))ᵀ M r^{k+1} ⊙ R^k / (φ_cur · φ_next)
```

where:
- `J_T(z)` = Jacobian of one PDHG step `T` at `z`
- `(I - J_T)ᵀ v` is computed via `update_primal_hyper_state!` / `update_dual_hyper_state!` using active-set masks
- `⊙` = elementwise multiply
- Division by `φ_cur · φ_next` normalizes the gradient

## Active-Set Masks

The Jacobian `J_T` involves indicator functions for active box constraints (primal) and active nonnegativity constraints (dual):

- Primal: `mask_x[i] = 1` if `l[i] < x[i] < u[i]` (interior of box), else `0`
- Dual: `mask_y[j] = 1` if `y[j] > 0` (strict interior of nonneg), else `0`

These masks are computed in `update_primal_hyper_state_kernel!` (line 488) and `update_dual_hyper_state_kernel!` (line 550).

## KKT Residual ρ

```
ρ = sqrt(ω · ‖primal_res‖² + (1/ω) · ‖dual_res‖² + |gap|²)
```

Implemented in `saddle_point_gpu.jl#compute_weight_kkt_residual` (line 250).
Uses `ConvergenceInformation` fields: `l2_primal_residual`, `l2_dual_residual`, `gap`.

## Primal Weight Update

```
ω_new = exp(s · log(‖Δy‖/‖Δx‖) + (1-s) · log(ω_old))
```

where `Δx = x_restart - x_last_restart`, `Δy = y_restart - y_last_restart`, `s = primal_weight_update_smoothing`.

Clipped to avoid degenerate weights. Stored in `CuPdhgSolverState.primal_weight`.
