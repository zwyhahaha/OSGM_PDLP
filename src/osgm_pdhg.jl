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
