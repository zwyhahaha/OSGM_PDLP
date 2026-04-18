# src/osgm_pdhg.jl

struct OsgmPdhgParameters
    osgm_stepsize::Float64   # η: OGD learning rate for preconditioner; 0.0 = plain PDHG
    osgm_block_size::Int64   # m: inner PDHG steps per outer OSGM step
end

mutable struct CuOsgmState
    z_start_primal::CuVector{Float64}         # z^k primal saved at block start
    z_start_dual::CuVector{Float64}           # z^k dual saved at block start
    block_endpoint_primal::CuVector{Float64}  # T^m(z^k) primal, saved before null-step
    block_endpoint_dual::CuVector{Float64}    # T^m(z^k) dual, saved before null-step
    primal_hyperparam::CuVector{Float64}      # diagonal preconditioner (primal), init ones
    dual_hyperparam::CuVector{Float64}        # diagonal preconditioner (dual), init ones
    phi_last::Float64                         # M-norm at z^k (carried from previous block)
end

mutable struct CuProbeBuffer
    delta_primal::CuVector{Float64}           # Δx from probe PDHG step
    delta_dual::CuVector{Float64}             # Δy from probe PDHG step (reused as dmrlam)
    delta_primal_product::CuVector{Float64}   # A Δx (free from primal kernel)
    delta_dual_product::CuVector{Float64}     # A^T Δy (extra SpMV in probe step)
    hyper_tmp::CuVector{Float64}              # scratch primal-size: mx → A^T dmrlam
    hyper_tmp2::CuVector{Float64}             # scratch dual-size: mlam → A A^T dmrlam
    aux_dual::CuVector{Float64}               # scratch dual-size: A mx → partial vlam
    ones_primal::CuVector{Float64}            # identity preconditioner for probe step
    ones_dual::CuVector{Float64}              # identity preconditioner for probe step
end

function initialize_osgm_state(primal_size::Int64, dual_size::Int64)
    return CuOsgmState(
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.ones(Float64, primal_size),
        CUDA.ones(Float64, dual_size),
        Inf,                                  # phi_last=Inf so first block always accepts
    )
end

function initialize_probe_buffer(primal_size::Int64, dual_size::Int64)
    return CuProbeBuffer(
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, dual_size),
        CUDA.ones(Float64, primal_size),
        CUDA.ones(Float64, dual_size),
    )
end

# ── Task 2: Preconditioned inner PDHG kernels ────────────────────────────────

function compute_next_primal_osgm_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    delta_primal::CuDeviceVector{Float64},
    primal_hyperparam::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            delta_primal[tx] = current_primal_solution[tx] -
                (step_size / primal_weight) * primal_hyperparam[tx] *
                (objective_vector[tx] - current_dual_product[tx])
            delta_primal[tx] = min(variable_upper_bound[tx],
                                   max(variable_lower_bound[tx], delta_primal[tx]))
            delta_primal[tx] -= current_primal_solution[tx]
        end
    end
    return
end

function compute_next_primal_osgm!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal::CuVector{Float64},
    delta_primal_product::CuVector{Float64},
    primal_hyperparam::CuVector{Float64},
)
    num_blocks = ceil(Int64, problem.num_variables / ThreadPerBlock)
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=num_blocks compute_next_primal_osgm_kernel!(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        current_primal_solution,
        current_dual_product,
        step_size,
        primal_weight,
        problem.num_variables,
        delta_primal,
        primal_hyperparam,
    )
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, delta_primal, 0,
                      delta_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function compute_next_dual_osgm_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    delta_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    extrapolation_coefficient::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    delta_dual::CuDeviceVector{Float64},
    dual_hyperparam::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    val = 0.0
    if tx <= num_constraints
        @inbounds begin
            val = current_dual_solution[tx] +
                (primal_weight * step_size) * dual_hyperparam[tx] *
                (right_hand_side[tx] -
                 (1 + extrapolation_coefficient) * delta_primal_product[tx] -
                 extrapolation_coefficient * current_primal_product[tx])
            if tx > num_equalities
                val = max(val, 0.0)
            end
            delta_dual[tx] = val - current_dual_solution[tx]
        end
    end
    return
end

function compute_next_dual_osgm!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    delta_dual::CuVector{Float64},
    dual_hyperparam::CuVector{Float64},
    extrapolation_coefficient::Float64 = 1.0,
)
    num_blocks = ceil(Int64, problem.num_constraints / ThreadPerBlock)
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=num_blocks compute_next_dual_osgm_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        delta_primal_product,
        step_size,
        primal_weight,
        extrapolation_coefficient,
        problem.num_equalities,
        problem.num_constraints,
        delta_dual,
        dual_hyperparam,
    )
end

function take_osgm_inner_step!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    osgm_state::CuOsgmState,
)
    compute_next_primal_osgm!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal,
        buffer_state.delta_primal_product,
        osgm_state.primal_hyperparam,
    )
    compute_next_dual_osgm!(
        problem,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal_product,
        solver_state.current_primal_product,
        buffer_state.delta_dual,
        osgm_state.dual_hyperparam,
    )
    solver_state.cumulative_kkt_passes += 1
    update_solution_in_solver_state!(problem, solver_state, buffer_state)
end

# ── Task 3: M-norm, z_cand, and Probe Step ──────────────────────────────────

function compute_m_norm(
    delta_primal::CuVector{Float64},
    delta_dual::CuVector{Float64},
    delta_primal_product::CuVector{Float64},
    primal_step_size::Float64,
    dual_step_size::Float64,
)
    norm_sq = CUDA.norm(delta_primal)^2 / primal_step_size +
              2.0 * CUDA.dot(delta_dual, delta_primal_product) +
              CUDA.norm(delta_dual)^2 / dual_step_size
    return sqrt(max(norm_sq, 0.0))
end

function compute_z_cand_kernel!(
    out::CuDeviceVector{Float64},
    z_start::CuDeviceVector{Float64},
    block_endpoint::CuDeviceVector{Float64},
    hyperparam::CuDeviceVector{Float64},
    n::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= n
        @inbounds begin
            R = z_start[tx] - block_endpoint[tx]
            out[tx] = z_start[tx] - hyperparam[tx] * R
        end
    end
    return
end

function compute_z_cand!(
    out::CuVector{Float64},
    z_start::CuVector{Float64},
    block_endpoint::CuVector{Float64},
    hyperparam::CuVector{Float64},
)
    n = length(out)
    num_blocks = ceil(Int64, n / ThreadPerBlock)
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=num_blocks compute_z_cand_kernel!(
        out, z_start, block_endpoint, hyperparam, n
    )
end

function recompute_a_products!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
)
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix,
                      solver_state.current_primal_solution, 0,
                      solver_state.current_primal_product, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t,
                      solver_state.current_dual_solution, 0,
                      solver_state.current_dual_product, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function run_probe_step!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    probe_buffer::CuProbeBuffer,
)
    step_size = solver_state.step_size
    primal_weight = solver_state.primal_weight

    compute_next_primal_osgm!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        step_size,
        primal_weight,
        probe_buffer.delta_primal,
        probe_buffer.delta_primal_product,
        probe_buffer.ones_primal,
    )
    compute_next_dual_osgm!(
        problem,
        solver_state.current_dual_solution,
        step_size,
        primal_weight,
        probe_buffer.delta_primal_product,
        solver_state.current_primal_product,
        probe_buffer.delta_dual,
        probe_buffer.ones_dual,
    )
    # A^T * delta_dual — needed for hypergradient mx computation
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t,
                      probe_buffer.delta_dual, 0,
                      probe_buffer.delta_dual_product, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

# ── Task 4: Hypergradient kernels and preconditioner OGD update ──────────────

function compute_mx_kernel!(
    hyper_tmp::CuDeviceVector{Float64},
    delta_primal::CuDeviceVector{Float64},
    delta_dual_product::CuDeviceVector{Float64},
    primal_step_size::Float64,
    n::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= n
        @inbounds hyper_tmp[tx] = delta_primal[tx] / primal_step_size + delta_dual_product[tx]
    end
    return
end

function compute_mlam_kernel!(
    hyper_tmp2::CuDeviceVector{Float64},
    delta_primal_product::CuDeviceVector{Float64},
    delta_dual::CuDeviceVector{Float64},
    dual_step_size::Float64,
    m::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= m
        @inbounds hyper_tmp2[tx] = delta_primal_product[tx] + delta_dual[tx] / dual_step_size
    end
    return
end

function compute_dmrlam_kernel!(
    dmrlam::CuDeviceVector{Float64},
    mlam::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    m::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= m
        @inbounds dmrlam[tx] = current_dual_solution[tx] > 0.0 ? mlam[tx] : 0.0
    end
    return
end

function compute_partial_vlam_kernel!(
    aux_dual::CuDeviceVector{Float64},
    mlam::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    primal_step_size::Float64,
    m::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= m
        @inbounds begin
            active = current_dual_solution[tx] > 0.0 ? 1.0 : 0.0
            aux_dual[tx] = -primal_step_size * aux_dual[tx] + (1.0 - active) * mlam[tx]
        end
    end
    return
end

function update_dual_hyperparam_kernel!(
    dual_hyperparam::CuDeviceVector{Float64},
    partial_vlam::CuDeviceVector{Float64},
    a_at_dmrlam::CuDeviceVector{Float64},
    z_start_dual::CuDeviceVector{Float64},
    block_endpoint_dual::CuDeviceVector{Float64},
    tau_p_tau_d::Float64,
    phi_prod::Float64,
    osgm_stepsize::Float64,
    m::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= m
        @inbounds begin
            R_dual = z_start_dual[tx] - block_endpoint_dual[tx]
            vlam = partial_vlam[tx] + 2.0 * tau_p_tau_d * a_at_dmrlam[tx]
            g = vlam * R_dual
            dual_hyperparam[tx] = max(0.0, dual_hyperparam[tx] + osgm_stepsize * g / phi_prod)
        end
    end
    return
end

function update_primal_hyperparam_kernel!(
    primal_hyperparam::CuDeviceVector{Float64},
    at_dmrlam::CuDeviceVector{Float64},
    z_start_primal::CuDeviceVector{Float64},
    block_endpoint_primal::CuDeviceVector{Float64},
    dual_step_size::Float64,
    phi_prod::Float64,
    osgm_stepsize::Float64,
    n::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= n
        @inbounds begin
            R_primal = z_start_primal[tx] - block_endpoint_primal[tx]
            g = dual_step_size * at_dmrlam[tx] * R_primal
            primal_hyperparam[tx] = max(0.0, primal_hyperparam[tx] + osgm_stepsize * g / phi_prod)
        end
    end
    return
end

function update_osgm_preconditioner!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_state::CuOsgmState,
    probe_buffer::CuProbeBuffer,
    phi_next::Float64,
    osgm_stepsize::Float64,
)
    phi_prod = osgm_state.phi_last * phi_next
    phi_prod < 1e-30 && return  # guard: skip if degenerate

    primal_step_size = solver_state.step_size / solver_state.primal_weight
    dual_step_size   = solver_state.step_size * solver_state.primal_weight
    tau_p_tau_d      = primal_step_size * dual_step_size  # = step_size^2

    n = problem.num_variables
    m = problem.num_constraints
    n_blocks = ceil(Int64, n / ThreadPerBlock)
    m_blocks = ceil(Int64, m / ThreadPerBlock)

    # 1. hyper_tmp = mx = delta_primal/τ_p + delta_dual_product
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=n_blocks compute_mx_kernel!(
        probe_buffer.hyper_tmp, probe_buffer.delta_primal, probe_buffer.delta_dual_product,
        primal_step_size, n)

    # 2. aux_dual = A * mx  (1 SpMV)
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix,
                      probe_buffer.hyper_tmp, 0, probe_buffer.aux_dual, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

    # 3. hyper_tmp2 = mlam = delta_primal_product + delta_dual/τ_d
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=m_blocks compute_mlam_kernel!(
        probe_buffer.hyper_tmp2, probe_buffer.delta_primal_product, probe_buffer.delta_dual,
        dual_step_size, m)

    # 4. delta_dual = dmrlam = active_dual ⊙ mlam  (reuse probe_buffer.delta_dual)
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=m_blocks compute_dmrlam_kernel!(
        probe_buffer.delta_dual, probe_buffer.hyper_tmp2,
        solver_state.current_dual_solution, m)

    # 5. hyper_tmp = A^T * dmrlam  (1 SpMV, overwrites mx)
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t,
                      probe_buffer.delta_dual, 0, probe_buffer.hyper_tmp, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

    # 6. aux_dual = -τ_p * (A mx) + (1-active) * mlam  (partial vlam)
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=m_blocks compute_partial_vlam_kernel!(
        probe_buffer.aux_dual, probe_buffer.hyper_tmp2,
        solver_state.current_dual_solution, primal_step_size, m)

    # 7. hyper_tmp2 = A * (A^T dmrlam)  (1 SpMV, overwrites mlam)
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix,
                      probe_buffer.hyper_tmp, 0, probe_buffer.hyper_tmp2, 'O',
                      CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

    # 8. Update dual_hyperparam
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=m_blocks update_dual_hyperparam_kernel!(
        osgm_state.dual_hyperparam, probe_buffer.aux_dual, probe_buffer.hyper_tmp2,
        osgm_state.z_start_dual, osgm_state.block_endpoint_dual,
        tau_p_tau_d, phi_prod, osgm_stepsize, m)

    # 9. Update primal_hyperparam
    CUDA.@sync @cuda threads=ThreadPerBlock blocks=n_blocks update_primal_hyperparam_kernel!(
        osgm_state.primal_hyperparam, probe_buffer.hyper_tmp,
        osgm_state.z_start_primal, osgm_state.block_endpoint_primal,
        dual_step_size, phi_prod, osgm_stepsize, n)
end

function osgm_boundary_step!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_state::CuOsgmState,
    probe_buffer::CuProbeBuffer,
    last_restart_info::CuRestartInfo,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    params::PdhgParameters,
    osgm_stepsize::Float64,
    buffer_avg::CuBufferAvgState,
    buffer_kkt::BufferKKTState,
    buffer_primal_gradient::CuVector{Float64},
    total_inner_iterations::Int64,
)
    step_size     = solver_state.step_size
    primal_weight = solver_state.primal_weight
    primal_step_size = step_size / primal_weight
    dual_step_size   = step_size * primal_weight

    # Step 4-5: compute z_cand in-place into current_primal/dual_solution
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

    # Step 6: probe at z_cand
    recompute_a_products!(problem, solver_state)
    run_probe_step!(problem, solver_state, probe_buffer)
    phi_trial = compute_m_norm(
        probe_buffer.delta_primal, probe_buffer.delta_dual,
        probe_buffer.delta_primal_product, primal_step_size, dual_step_size)

    # Step 7: null-step decision
    if phi_trial <= osgm_state.phi_last
        # ACCEPT z_cand: current state stays as z_cand
        phi_next = phi_trial
    else
        # REJECT: restore T^m(z^k) and probe there
        solver_state.current_primal_solution .= osgm_state.block_endpoint_primal
        solver_state.current_dual_solution   .= osgm_state.block_endpoint_dual
        recompute_a_products!(problem, solver_state)
        run_probe_step!(problem, solver_state, probe_buffer)
        phi_next = compute_m_norm(
            probe_buffer.delta_primal, probe_buffer.delta_dual,
            probe_buffer.delta_primal_product, primal_step_size, dual_step_size)
    end

    # Step 8: restart scheme
    restart_used = run_restart_scheme(
        problem,
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        last_restart_info,
        total_inner_iterations,
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

    # Step 9: primal weight update on restart
    if restart_used != RESTART_CHOICE_NO_RESTART
        solver_state.primal_weight = compute_new_primal_weight(
            last_restart_info,
            solver_state.primal_weight,
            params.restart_params.primal_weight_update_smoothing,
            params.verbosity,
        )
        solver_state.ratio_step_sizes = 1.0
    end

    # Step 10: update preconditioner
    if osgm_stepsize > 0.0
        update_osgm_preconditioner!(
            problem, solver_state, osgm_state, probe_buffer, phi_next, osgm_stepsize)
    end

    # Step 11: carry phi_last forward
    osgm_state.phi_last = phi_next

    return restart_used
end
