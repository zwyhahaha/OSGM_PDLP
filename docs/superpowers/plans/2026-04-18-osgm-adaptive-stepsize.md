# OSGM Adaptive Stepsize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-block OSGM with event-driven adaptive-stepsize PDHG where the OSGM preconditioner update fires only on adaptive restart.

**Architecture:** The outer OSGM loop is replaced by a plain per-step adaptive PDHG loop (identical to `primal_dual_hybrid_gradient_gpu.jl`). At each termination evaluation point, `osgm_boundary_step!` computes z_cand, calls `run_restart_scheme`, and — only if restart fires — updates the hypergradient and saves `z_start`/`phi_last`. The M-norm is fixed to `ref_step_size = (1-ε)/σ_max(A)` computed once from the power method.

**Tech Stack:** Julia, CUDA.jl, cuSPARSE. All changes are in `src/osgm_pdhg.jl`, `scripts/solve.jl`, and `test/osgm_sanity_check.jl`.

---

## File Map

| File | What changes |
|---|---|
| `src/osgm_pdhg.jl` | All algorithmic changes (structs, inner step, probe, boundary, main loop) |
| `scripts/solve.jl` | Remove `--osgm_block_size` CLI arg; update constructor call |
| `test/osgm_sanity_check.jl` | Update to `AdaptiveStepsizeParams`; remove `osgm_block_size` |

---

### Task 1: Simplify `OsgmPdhgParameters`

**Files:**
- Modify: `src/osgm_pdhg.jl:3-6`

- [ ] **Step 1: Replace the struct definition**

Old:
```julia
struct OsgmPdhgParameters
    osgm_stepsize::Float64
    osgm_block_size::Int64
end
```

New:
```julia
struct OsgmPdhgParameters
    osgm_stepsize::Float64
end
```

- [ ] **Step 2: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "refactor: remove osgm_block_size from OsgmPdhgParameters"
```

---

### Task 2: Adaptive inner step — `take_osgm_inner_step!`

**Files:**
- Modify: `src/osgm_pdhg.jl:167-195`

`take_osgm_inner_step!` currently takes one fixed step and always commits. Replace with the same interaction/movement backtracking loop used in `take_step!(AdaptiveStepsizeParams, ...)` in `primal_dual_hybrid_gradient_gpu.jl`, but calling the OSGM primal/dual kernels.

Note: `compute_interaction_and_movement` and `update_solution_in_solver_state!` are defined in `primal_dual_hybrid_gradient_gpu.jl` and are accessible within the same module.

- [ ] **Step 1: Replace `take_osgm_inner_step!`**

```julia
function take_osgm_inner_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    osgm_state::CuOsgmState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        compute_next_primal_osgm!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal,
            buffer_state.delta_primal_product,
            osgm_state.primal_hyperparam,
        )
        compute_next_dual_osgm!(
            problem,
            solver_state.current_dual_solution,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal_product,
            solver_state.current_primal_product,
            buffer_state.delta_dual,
            osgm_state.dual_hyperparam,
        )

        interaction, movement = compute_interaction_and_movement(
            solver_state, problem, buffer_state)

        solver_state.cumulative_kkt_passes += 1

        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        if step_size <= step_size_limit
            update_solution_in_solver_state!(problem, solver_state, buffer_state)
            done = true
        end

        first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(step_params.reduction_exponent)) * step_size_limit
        second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(step_params.growth_exponent)) * step_size
        step_size = min(first_term, second_term)
    end
    solver_state.step_size = step_size
end
```

- [ ] **Step 2: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "feat: add adaptive backtracking to take_osgm_inner_step!"
```

---

### Task 3: Fix `run_probe_step!` to use `ref_step_size`

**Files:**
- Modify: `src/osgm_pdhg.jl:254-287`

The probe computes the single-step residual `r = z - T(z)` used for the M-norm. It must use the fixed `ref_step_size`, not the drifting `solver_state.step_size`.

- [ ] **Step 1: Add `ref_step_size` parameter and use it**

```julia
function run_probe_step!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    probe_buffer::CuProbeBuffer,
    ref_step_size::Float64,
)
    primal_weight = solver_state.primal_weight

    compute_next_primal_osgm!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        ref_step_size,
        primal_weight,
        probe_buffer.delta_primal,
        probe_buffer.delta_primal_product,
        probe_buffer.ones_primal,
    )
    compute_next_dual_osgm!(
        problem,
        solver_state.current_dual_solution,
        ref_step_size,
        primal_weight,
        probe_buffer.delta_primal_product,
        solver_state.current_primal_product,
        probe_buffer.delta_dual,
        probe_buffer.ones_dual,
    )
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t,
                      probe_buffer.delta_dual, 0,
                      probe_buffer.delta_dual_product, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end
```

- [ ] **Step 2: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "feat: run_probe_step! uses ref_step_size for stable M-norm"
```

---

### Task 4: Fix `update_osgm_preconditioner!` to use `ref_step_size`

**Files:**
- Modify: `src/osgm_pdhg.jl:391-457`

Lines 402–404 derive `primal_step_size` and `dual_step_size` from `solver_state.step_size`. Replace with `ref_step_size`. Also update the two hyperparam kernel calls (lines 447–456) to use `osgm_state.block_endpoint_primal/dual` — these are still correct since `osgm_boundary_step!` saves T^m(z^k) there before computing z_cand.

- [ ] **Step 1: Add `ref_step_size` parameter; replace step_size derivation**

Change the function signature and lines 402–404:

```julia
function update_osgm_preconditioner!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_state::CuOsgmState,
    probe_buffer::CuProbeBuffer,
    phi_next::Float64,
    osgm_stepsize::Float64,
    ref_step_size::Float64,
)
    phi_prod = osgm_state.phi_last * phi_next
    phi_prod < 1e-30 && return

    primal_step_size = ref_step_size / solver_state.primal_weight
    dual_step_size   = ref_step_size * solver_state.primal_weight
    tau_p_tau_d      = primal_step_size * dual_step_size
```

The rest of the function body (the 9 SpMV/kernel steps) is unchanged.

- [ ] **Step 2: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "feat: update_osgm_preconditioner! uses ref_step_size for Jacobian"
```

---

### Task 5: Rewrite `osgm_boundary_step!`

**Files:**
- Modify: `src/osgm_pdhg.jl:459-556`

This function is called at each termination evaluation point. It:
1. Saves T^m(z^k) into `block_endpoint` (still needed for hypergradient kernels)
2. Computes z_cand in-place
3. Calls `run_restart_scheme` with z_cand as "current"
4. If restart fired: probes phi_next, updates hypergradient, saves z_start/phi_last

`primal_norm_params` and `dual_norm_params` are computed from `ref_step_size` (not `solver_state.step_size`) so the KKT metric is consistent.

- [ ] **Step 1: Replace the entire function**

```julia
function osgm_boundary_step!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_state::CuOsgmState,
    probe_buffer::CuProbeBuffer,
    last_restart_info::CuRestartInfo,
    params::PdhgParameters,
    osgm_stepsize::Float64,
    ref_step_size::Float64,
    buffer_avg::CuBufferAvgState,
    buffer_kkt::BufferKKTState,
    buffer_primal_gradient::CuVector{Float64},
    total_iterations::Int64,
)
    primal_norm_params = ref_step_size / solver_state.primal_weight
    dual_norm_params   = ref_step_size * solver_state.primal_weight

    # Save T^m(z^k) — needed by hypergradient kernels
    osgm_state.block_endpoint_primal .= solver_state.current_primal_solution
    osgm_state.block_endpoint_dual   .= solver_state.current_dual_solution

    # Compute z_cand in-place (P_k = I when osgm_stepsize=0 → z_cand = current)
    compute_z_cand!(
        solver_state.current_primal_solution,
        osgm_state.z_start_primal,
        osgm_state.block_endpoint_primal,
        osgm_state.primal_hyperparam,
    )
    compute_z_cand!(
        solver_state.current_dual_solution,
        osgm_state.z_start_dual,
        osgm_state.block_endpoint_dual,
        osgm_state.dual_hyperparam,
    )
    recompute_a_products!(problem, solver_state)

    # Restart scheme: z_cand is now "current"; picks z_cand or avg
    restart_used = run_restart_scheme(
        problem,
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        last_restart_info,
        total_iterations,
        primal_norm_params,
        dual_norm_params,
        solver_state.primal_weight,
        params.verbosity,
        params.restart_params,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        buffer_avg,
        buffer_kkt,
        buffer_primal_gradient,
    )

    if restart_used != RESTART_CHOICE_NO_RESTART
        # Primal weight update
        solver_state.primal_weight = compute_new_primal_weight(
            last_restart_info,
            solver_state.primal_weight,
            params.restart_params.primal_weight_update_smoothing,
            params.verbosity,
        )
        solver_state.ratio_step_sizes = 1.0

        # Probe at accepted point → phi_next (uses updated primal_weight)
        run_probe_step!(problem, solver_state, probe_buffer, ref_step_size)
        primal_step_size = ref_step_size / solver_state.primal_weight
        dual_step_size   = ref_step_size * solver_state.primal_weight
        phi_next = compute_m_norm(
            probe_buffer.delta_primal, probe_buffer.delta_dual,
            probe_buffer.delta_primal_product, primal_step_size, dual_step_size)

        # Hypergradient update
        if osgm_stepsize > 0.0
            update_osgm_preconditioner!(
                problem, solver_state, osgm_state, probe_buffer,
                phi_next, osgm_stepsize, ref_step_size)
        end

        # Carry forward for next block
        osgm_state.phi_last = phi_next
        osgm_state.z_start_primal .= solver_state.current_primal_solution
        osgm_state.z_start_dual   .= solver_state.current_dual_solution
    end

    return restart_used
end
```

- [ ] **Step 2: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "feat: rewrite osgm_boundary_step! — restart-driven OSGM update"
```

---

### Task 6: Rewrite the `optimize` main loop

**Files:**
- Modify: `src/osgm_pdhg.jl:617-868`

Replace the outer OSGM loop with a plain per-step loop. Key differences from old code:

- `ref_step_size` is a local variable (power method result); `solver_state.step_size` starts here and drifts
- The step-size policy from `params.step_size_policy_params` is passed to `take_osgm_inner_step!` (must be `AdaptiveStepsizeParams`)
- `total_inner_iters` = `solver_state.total_number_iterations` (tracked inside the adaptive step)
- At each termination evaluation (`mod(iteration-1, termination_evaluation_frequency) == 0`): update avg, check termination, call `osgm_boundary_step!`
- `take_step!(osgm_params, ...)` is removed; replaced by `take_osgm_inner_step!` per iteration

- [ ] **Step 1: Delete `take_step!(osgm_params, ...)` (lines 558–615)**

Remove the entire `function take_step!(osgm_params::OsgmPdhgParameters, ...)` block.

- [ ] **Step 2: Replace the main loop inside `optimize`**

Replace everything from `outer_iteration = 0` (line 761) through `end` (line 868) with:

```julia
    # Extract step policy — must be AdaptiveStepsizeParams for OSGM
    step_params = params.step_size_policy_params

    iteration = 0

    while true
        iteration += 1

        if mod(iteration - 1, params.termination_evaluation_frequency) == 0 ||
            solver_state.numerical_error

            total_inner_iters = solver_state.total_number_iterations
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            if solver_state.solution_weighted_avg.sum_primal_solutions_count == 0
                buffer_avg.avg_primal_solution .= solver_state.current_primal_solution
                buffer_avg.avg_dual_solution   .= solver_state.current_dual_solution
                buffer_avg.avg_primal_product  .= solver_state.current_primal_product
                buffer_avg.avg_primal_gradient .= buffer_primal_gradient
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                total_inner_iters,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE,
                buffer_avg.avg_primal_product,
                buffer_avg.avg_primal_gradient,
                buffer_original,
                buffer_kkt,
                buffer_kkt_infeas,
                buffer_lp,
            )
            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] = time_spent_doing_basic_algorithm

            termination_reason = check_termination_criteria(
                termination_criteria, qp_cache, current_iteration_stats)

            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            if total_inner_iters < 10 && (
                termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE ||
                termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
                termination_reason = false
            end

            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            if print_to_screen_this_iteration(
                termination_reason, total_inner_iters, params.verbosity,
                Int32(params.termination_evaluation_frequency))
                display_iteration_stats(current_iteration_stats, params.verbosity)
            end

            if termination_reason != false
                avg_primal_solution = zeros(primal_size)
                avg_dual_solution   = zeros(dual_size)
                gpu_to_cpu!(buffer_avg.avg_primal_solution, buffer_avg.avg_dual_solution,
                            avg_primal_solution, avg_dual_solution)

                pdhg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    total_inner_iters,
                    termination_reason,
                    current_iteration_stats)

                return unscaled_saddle_point_output(
                    scaled_problem,
                    avg_primal_solution,
                    avg_dual_solution,
                    termination_reason,
                    total_inner_iters - 1,
                    iteration_stats)
            end

            t0 = time()
            buffer_primal_gradient .= d_problem.objective_vector .- solver_state.current_dual_product
            osgm_boundary_step!(
                d_problem,
                solver_state,
                osgm_state,
                probe_buffer,
                last_restart_info,
                params,
                osgm_params.osgm_stepsize,
                ref_step_size,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
                total_inner_iters,
            )
            time_spent_doing_basic_algorithm += time() - t0
        end

        take_osgm_inner_step!(step_params, d_problem, solver_state, inner_buffer, osgm_state)
    end
```

Also update the `solver_state.step_size` initialization section (lines 669–676) to store the power method result in a local `ref_step_size`:

```julia
    # Compute ref_step_size from power method — used for M-norm throughout
    desired_relative_error = 1e-6
    maximum_singular_value, num_power_iters = estimate_maximum_singular_value(
        scaled_problem.scaled_qp.constraint_matrix,
        probability_of_failure = 0.001,
        desired_relative_error = desired_relative_error,
    )
    ref_step_size = (1 - desired_relative_error) / maximum_singular_value
    solver_state.step_size = ref_step_size  # adaptive steps start here and drift
    solver_state.cumulative_kkt_passes += num_power_iters
```

- [ ] **Step 3: Verify the file compiles**

```bash
CUDA_VISIBLE_DEVICES=1 julia --project -e 'import cuPDLP; println("OK")'
```

Expected: `OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/osgm_pdhg.jl
git commit -m "feat: replace fixed OSGM outer loop with per-step adaptive loop"
```

---

### Task 7: Update `scripts/solve.jl`

**Files:**
- Modify: `scripts/solve.jl`

Remove `--osgm_block_size` arg and update the `OsgmPdhgParameters` constructor call.

- [ ] **Step 1: Remove `--osgm_block_size` from `parse_command_line`**

Delete these lines (around line 155–159):

```julia
        "--osgm_block_size"
        help = "OSGM block size m (number of inner PDHG steps per outer OSGM step)."
        arg_type = Int
        default = 64
```

- [ ] **Step 2: Remove `osgm_block_size` from `main`**

Delete (around line 172):
```julia
    osgm_block_size = parsed_args["osgm_block_size"]
```

- [ ] **Step 3: Update `OsgmPdhgParameters` constructor**

Change (around line 218):
```julia
    cuPDLP.OsgmPdhgParameters(osgm_stepsize, osgm_block_size),
```
to:
```julia
    cuPDLP.OsgmPdhgParameters(osgm_stepsize),
```

- [ ] **Step 4: Commit**

```bash
git add scripts/solve.jl
git commit -m "refactor: remove --osgm_block_size CLI arg from solve.jl"
```

---

### Task 8: Update `test/osgm_sanity_check.jl`

**Files:**
- Modify: `test/osgm_sanity_check.jl`

With `osgm_stepsize=0` and `P_k=I`: `z_cand = z_start - I ⊙ (z_start - current) = current`. So `osgm_boundary_step!` passes the unmodified current iterate to `run_restart_scheme`. This is identical to what plain PDHG does. The sanity check must use `AdaptiveStepsizeParams` for plain PDHG (since OSGM now always uses adaptive steps).

- [ ] **Step 1: Rewrite the test**

```julia
using cuPDLP
using Test

@testset "OSGM sanity check: osgm_stepsize=0 matches plain PDHG on afiro" begin
    instance_path = joinpath(@__DIR__, "..", "data", "netlib", "afiro.mps.gz")
    @test isfile(instance_path)

    lp = cuPDLP.qps_reader_to_standard_form(instance_path)

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,
        cuPDLP.KKT_GREEDY,
        1000, 0.36, 0.2, 0.8, 0.5,
    )
    termination_params = cuPDLP.construct_termination_criteria(
        eps_optimal_absolute = 1e-4,
        eps_optimal_relative = 1e-4,
        eps_primal_infeasible = 1e-8,
        eps_dual_infeasible = 1e-8,
        time_sec_limit = 60.0,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )
    params = cuPDLP.PdhgParameters(
        10, false, 1.0, 1.0, true, 0, true, 64,
        termination_params, restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3, 0.6),
    )

    # Plain PDHG (no OSGM) — uses AdaptiveStepsizeParams
    result_pdhg = cuPDLP.optimize(params, lp)

    # OSGM with stepsize=0 — z_cand = current always, so identical to plain PDHG
    result_osgm = cuPDLP.optimize(params, cuPDLP.OsgmPdhgParameters(0.0), lp)

    @test result_pdhg.termination_reason == result_osgm.termination_reason
    @test abs(result_pdhg.iteration_count - result_osgm.iteration_count) <= 64

    pdhg_obj = result_pdhg.iteration_stats[end].convergence_information[1].primal_objective
    osgm_obj = result_osgm.iteration_stats[end].convergence_information[1].primal_objective
    @test abs(pdhg_obj - osgm_obj) < 2e-2
end
```

- [ ] **Step 2: Commit**

```bash
git add test/osgm_sanity_check.jl
git commit -m "test: update osgm_sanity_check for adaptive stepsize and no block_size"
```

---

### Task 9: Run sanity check

**Files:** none

- [ ] **Step 1: Run the sanity check (must finish in < 60 s)**

```bash
CUDA_VISIBLE_DEVICES=1 julia --project test/osgm_sanity_check.jl
```

Expected output: `Test Summary: ... 4 passed` with no failures. If it hangs beyond 60 s there is a bug — likely an infinite loop in the adaptive backtracking or a broken convergence check.

- [ ] **Step 2: Run a live OSGM solve on afiro to confirm it completes**

```bash
CUDA_VISIBLE_DEVICES=1 julia --project scripts/solve.jl \
  --instance_path data/netlib/afiro.mps.gz \
  --output_directory results/netlib \
  --tolerance 1e-4 \
  --time_sec_limit 60 \
  --osgm_stepsize 0.5
```

Expected: terminates in under 60 s with a valid solution. If it takes longer, investigate — likely the backtracking is misconfigured or restart never fires.

- [ ] **Step 3: Commit any fixes, then tag**

```bash
git add -p  # stage only intentional changes
git commit -m "fix: <describe what was wrong>"
```
