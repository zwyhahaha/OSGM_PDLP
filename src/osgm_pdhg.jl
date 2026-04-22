struct AdaptiveStepsizeParams
    reduction_exponent::Float64
    growth_exponent::Float64
end

struct ConstantStepsizeParams end

mutable struct PdhgParameters
    l_inf_ruiz_iterations::Int
    l2_norm_rescaling::Bool
    pock_chambolle_alpha::Union{Float64,Nothing}
    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        AdaptiveStepsizeParams,
        ConstantStepsizeParams,
    }
end

mutable struct CuPdhgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    solution_weighted_avg::CuSolutionWeightedAverage
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
end

mutable struct CuBufferState
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    delta_primal_product::CuVector{Float64}
end

struct OsgmPdhgParameters
    osgm_stepsize::Float64
    osgm_blocksize::Int
    use_null_step::Bool
end

const OsgmParams = OsgmPdhgParameters

mutable struct OsgmSolverState
    primal_preconditioner::CuVector{Float64}
    dual_preconditioner::CuVector{Float64}
    block_primal_solution::CuVector{Float64}
    block_dual_solution::CuVector{Float64}
    block_primal_product::CuVector{Float64}
    block_dual_product::CuVector{Float64}
    tau_ref::Float64
    sigma_ref::Float64
    accepted_steps_in_block::Int64
    delta_block_primal::CuVector{Float64}
    delta_block_dual::CuVector{Float64}
    candidate_primal_solution::CuVector{Float64}
    candidate_dual_solution::CuVector{Float64}
    candidate_primal_product::CuVector{Float64}
    candidate_dual_product::CuVector{Float64}
    reference_primal_solution::CuVector{Float64}
    reference_dual_solution::CuVector{Float64}
    residual_primal::CuVector{Float64}
    residual_dual::CuVector{Float64}
    primal_mask::CuVector{Float64}
    dual_mask::CuVector{Float64}
    metric_primal::CuVector{Float64}
    metric_dual::CuVector{Float64}
    gradient_primal::CuVector{Float64}
    gradient_dual::CuVector{Float64}
    tmp_primal::CuVector{Float64}
    tmp_dual::CuVector{Float64}
end

function define_norms(
    primal_size::Int64,
    dual_size::Int64,
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end
  

function pdhg_specific_log(
    # problem::QuadraticProgrammingProblem,
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        # "   %5d inv_step_size=%9g ",
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        CUDA.norm(current_primal_solution),
        CUDA.norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

function pdhg_final_log(
    problem::QuadraticProgrammingProblem,
    avg_primal_solution::Vector{Float64},
    avg_dual_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        # infeas = max_primal_violation(problem, avg_primal_solution)
        # primal_obj_val = primal_obj(problem, avg_primal_solution)
        # dual_stats =
        #     compute_dual_stats(problem, avg_primal_solution, avg_dual_solution)
        
        println("Avg solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_primal_solution, 1),
            CUDA.norm(avg_primal_solution),
            CUDA.norm(avg_primal_solution, Inf)
        )
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_dual_solution, 1),
            CUDA.norm(avg_dual_solution),
            CUDA.norm(avg_dual_solution, Inf)
        )
    end

    generic_final_log(
        problem,
        avg_primal_solution,
        avg_dual_solution,
        last_iteration_stats,
        verbosity,
        iteration,
        termination_reason,
    )
end

function power_method_failure_probability(
    dimension::Int64,
    epsilon::Float64,
    k::Int64,
)
    if k < 2 || epsilon <= 0.0
        return 1.0
    end
    return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) * sqrt(dimension) * (1.0 - epsilon)^(k - 1 / 2) # FirstOrderLp.jl old version (epsilon * (k - 1)) instead of sqrt(epsilon * (k - 1)))
end

function estimate_maximum_singular_value(
    matrix::SparseMatrixCSC{Float64,Int64};
    probability_of_failure = 0.01::Float64,
    desired_relative_error = 0.1::Float64,
    seed::Int64 = 1,
)
    epsilon = 1.0 - (1.0 - desired_relative_error)^2
    x = randn(Random.MersenneTwister(seed), size(matrix, 2))

    number_of_power_iterations = 0
    while power_method_failure_probability(
        size(matrix, 2),
        epsilon,
        number_of_power_iterations,
    ) > probability_of_failure
        x = x / norm(x, 2)
        x = matrix' * (matrix * x)
        number_of_power_iterations += 1
    end
    
    return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
    number_of_power_iterations
end

"""
Kernel to compute primal solution in the next iteration
"""
function compute_next_primal_solution_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    delta_primal::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            delta_primal[tx] = current_primal_solution[tx] - (step_size / primal_weight) * (objective_vector[tx] - current_dual_product[tx])
            delta_primal[tx] = min(variable_upper_bound[tx], max(variable_lower_bound[tx], delta_primal[tx]))
            delta_primal[tx] -= current_primal_solution[tx]
        end
    end
    return 
end

"""
Compute primal solution in the next iteration
"""
function compute_next_primal_solution!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal::CuVector{Float64},
    delta_primal_product::CuVector{Float64},
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_next_primal_solution_kernel!(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        current_primal_solution,
        current_dual_product,
        step_size,
        primal_weight,
        problem.num_variables,
        delta_primal,
    )

    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, delta_primal, 0, delta_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    
end

"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
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
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])

            delta_dual[tx] -= current_dual_solution[tx]
        end
    elseif num_equalities + 1 <= tx <= num_constraints
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])
            delta_dual[tx] = max(delta_dual[tx], 0.0)

            delta_dual[tx] -= current_dual_solution[tx]
        end
    end
    return 
end

"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    delta_dual::CuVector{Float64},
    extrapolation_coefficient::Float64 = 1.0,
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
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
    )

    # next_dual_product .= problem.constraint_matrix_t * next_dual
    # CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    # solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    solver_state.current_primal_solution .+= buffer_state.delta_primal
    solver_state.current_primal_product .+= buffer_state.delta_primal_product

    solver_state.current_dual_solution .+= buffer_state.delta_dual
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, solver_state.current_dual_solution, 0, solver_state.current_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    
    weight = solver_state.step_size
    
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
    )
end

function estimate_reference_step_size(
    matrix::SparseMatrixCSC{Float64,Int64},
)
    desired_relative_error = 0.2
    maximum_singular_value, number_of_power_iterations =
        estimate_maximum_singular_value(
            matrix,
            probability_of_failure = 0.001,
            desired_relative_error = desired_relative_error,
        )
    return (1 - desired_relative_error) / maximum_singular_value,
    number_of_power_iterations
end

function osgm_primal_matvec!(
    problem::CuLinearProgrammingProblem,
    input::CuVector{Float64},
    output::CuVector{Float64},
    solver_state::CuPdhgSolverState,
)
    CUDA.CUSPARSE.mv!(
        'N',
        1,
        problem.constraint_matrix,
        input,
        0,
        output,
        'O',
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2,
    )
    solver_state.cumulative_kkt_passes += 1
end

function osgm_dual_matvec!(
    problem::CuLinearProgrammingProblem,
    input::CuVector{Float64},
    output::CuVector{Float64},
    solver_state::CuPdhgSolverState,
)
    CUDA.CUSPARSE.mv!(
        'N',
        1,
        problem.constraint_matrix_t,
        input,
        0,
        output,
        'O',
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2,
    )
    solver_state.cumulative_kkt_passes += 1
end

function sync_osgm_block_start!(
    solver_state::CuPdhgSolverState,
    osgm_solver_state::OsgmSolverState,
)
    # Record z_blk = current z at the start of each block (spec §1, §8).
    # Called at initialization and after every restart or osgm_update!.
    osgm_solver_state.block_primal_solution .= solver_state.current_primal_solution
    osgm_solver_state.block_dual_solution .= solver_state.current_dual_solution
    osgm_solver_state.block_primal_product .= solver_state.current_primal_product
    osgm_solver_state.block_dual_product .= solver_state.current_dual_product
    osgm_solver_state.accepted_steps_in_block = 0
end

function reference_pdhg_residual!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_solver_state::OsgmSolverState,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    dual_product::CuVector{Float64},  # = A^T * dual_solution (pre-computed)
)
    # Computes the reference one-step PDHG map T(z) and residual r(z) = z - T(z)
    # using fixed stepsizes (tau_ref, sigma_ref). Also records D_x, D_y masks
    # (Jacobians of box/orthant projections) for gradient computation (spec §2, §6).

    # x_ref = Pi_X(x - tau_ref * (c - A^T y)); D_x = 1 where interior of box.
    osgm_solver_state.reference_primal_solution .=
        primal_solution .-
        osgm_solver_state.tau_ref .* (problem.objective_vector .- dual_product)
    osgm_solver_state.primal_mask .= ifelse.(
        (osgm_solver_state.reference_primal_solution .> problem.variable_lower_bound) .&
        (osgm_solver_state.reference_primal_solution .< problem.variable_upper_bound),
        1.0,
        0.0,
    )
    osgm_solver_state.reference_primal_solution .= min.(
        problem.variable_upper_bound,
        max.(
            problem.variable_lower_bound,
            osgm_solver_state.reference_primal_solution,
        ),
    )

    # y_ref = Pi_Y(y + sigma_ref * (b - A(2 x_ref - x))); D_y = 1 where y_ref > 0 or equality row.
    osgm_solver_state.tmp_primal .=
        2.0 .* osgm_solver_state.reference_primal_solution .- primal_solution
    osgm_primal_matvec!(
        problem,
        osgm_solver_state.tmp_primal,
        osgm_solver_state.tmp_dual,
        solver_state,
    )

    osgm_solver_state.reference_dual_solution .=
        dual_solution .+
        osgm_solver_state.sigma_ref .* (problem.right_hand_side .- osgm_solver_state.tmp_dual)
    osgm_solver_state.dual_mask .= 0.0
    if problem.num_equalities > 0
        osgm_solver_state.dual_mask[1:problem.num_equalities] .= 1.0
    end
    if problem.num_equalities < problem.num_constraints
        inequality_range = (problem.num_equalities + 1):problem.num_constraints
        osgm_solver_state.reference_dual_solution[inequality_range] .= max.(
            osgm_solver_state.reference_dual_solution[inequality_range],
            0.0,
        )
        osgm_solver_state.dual_mask[inequality_range] .= ifelse.(
            osgm_solver_state.reference_dual_solution[inequality_range] .> 0.0,
            1.0,
            0.0,
        )
    end

    # r(z) = z - T(z)
    osgm_solver_state.residual_primal .=
        primal_solution .- osgm_solver_state.reference_primal_solution
    osgm_solver_state.residual_dual .=
        dual_solution .- osgm_solver_state.reference_dual_solution
end

function residual_metric_norm_squared(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_solver_state::OsgmSolverState,
)
    # Computes ||r||_M^2 = r^T M r where M = [[tau^-1 I, A^T], [A, sigma^-1 I]] (spec §2).
    # Expands to: ||r_x||^2/tau + ||r_y||^2/sigma + 2 r_y^T A r_x.
    osgm_primal_matvec!(
        problem,
        osgm_solver_state.residual_primal,
        osgm_solver_state.tmp_dual,   # tmp_dual = A r_x
        solver_state,
    )
    primal_residual_norm_sq =
        CUDA.dot(
            osgm_solver_state.residual_primal,
            osgm_solver_state.residual_primal,
        )
    dual_residual_norm_sq =
        CUDA.dot(
            osgm_solver_state.residual_dual,
            osgm_solver_state.residual_dual,
        )
    interaction =
        CUDA.dot(osgm_solver_state.residual_dual, osgm_solver_state.tmp_dual)  # r_y^T A r_x
    return primal_residual_norm_sq / osgm_solver_state.tau_ref +
           dual_residual_norm_sq / osgm_solver_state.sigma_ref +
           2.0 * interaction
end

function apply_metric_M!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_solver_state::OsgmSolverState,
)
    # Computes v = M r, i.e. v_x = A^T r_y + r_x/tau, v_y = A r_x + r_y/sigma (spec §7).
    # Stores result as (metric_primal, metric_dual) = (v_x, v_y).
    osgm_dual_matvec!(
        problem,
        osgm_solver_state.residual_dual,
        osgm_solver_state.metric_primal,   # A^T r_y
        solver_state,
    )
    osgm_solver_state.metric_primal .=
        osgm_solver_state.metric_primal .+
        osgm_solver_state.residual_primal ./ osgm_solver_state.tau_ref  # v_x = A^T r_y + r_x/tau

    osgm_primal_matvec!(
        problem,
        osgm_solver_state.residual_primal,
        osgm_solver_state.metric_dual,     # A r_x
        solver_state,
    )
    osgm_solver_state.metric_dual .=
        osgm_solver_state.metric_dual .+
        osgm_solver_state.residual_dual ./ osgm_solver_state.sigma_ref  # v_y = A r_x + r_y/sigma
end

function compute_osgm_gradient!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    osgm_solver_state::OsgmSolverState,
)
    # Computes g = (I - J_T(z_osgm))^T M r(z_osgm) using the efficient formula (spec §7).
    # Assumes metric_primal = v_x and metric_dual = v_y have already been set by apply_metric_M!.
    # NOTE: overwrites residual_primal and residual_dual as scratch space — callers must not
    # use those buffers for r(z_osgm) after this call.

    # g_x = (I - D_x) v_x - sigma * (I - 2 D_x) A^T D_y v_y
    osgm_solver_state.tmp_dual .=
        osgm_solver_state.dual_mask .* osgm_solver_state.metric_dual   # D_y v_y
    osgm_dual_matvec!(
        problem,
        osgm_solver_state.tmp_dual,
        osgm_solver_state.tmp_primal,   # A^T D_y v_y
        solver_state,
    )
    osgm_solver_state.gradient_primal .=
        (1.0 .- osgm_solver_state.primal_mask) .* osgm_solver_state.metric_primal .-
        osgm_solver_state.sigma_ref .* (1.0 .- 2.0 .* osgm_solver_state.primal_mask) .*
        osgm_solver_state.tmp_primal

    # g_y = -tau * A D_x v_x + (I - D_y) v_y + 2 tau sigma A D_x A^T D_y v_y
    # Uses reference_primal_solution and reference_dual_solution as temporaries.
    osgm_solver_state.reference_primal_solution .=
        osgm_solver_state.primal_mask .* osgm_solver_state.metric_primal  # D_x v_x
    osgm_primal_matvec!(
        problem,
        osgm_solver_state.reference_primal_solution,
        osgm_solver_state.reference_dual_solution,  # A D_x v_x
        solver_state,
    )
    osgm_solver_state.gradient_dual .=
        .-osgm_solver_state.tau_ref .* osgm_solver_state.reference_dual_solution .+
        (1.0 .- osgm_solver_state.dual_mask) .* osgm_solver_state.metric_dual  # partial g_y

    osgm_solver_state.residual_primal .=
        osgm_solver_state.primal_mask .* osgm_solver_state.tmp_primal   # D_x A^T D_y v_y
    osgm_primal_matvec!(
        problem,
        osgm_solver_state.residual_primal,
        osgm_solver_state.residual_dual,   # A D_x A^T D_y v_y
        solver_state,
    )
    osgm_solver_state.gradient_dual .+=
        2.0 * osgm_solver_state.tau_ref * osgm_solver_state.sigma_ref .*
        osgm_solver_state.residual_dual   # completes g_y
end

"""
Compute iteraction and movement for AdaptiveStepsize
"""
function compute_interaction_and_movement(
    solver_state::CuPdhgSolverState,
    problem::CuLinearProgrammingProblem,
    buffer_state::CuBufferState,
)
    primal_dual_interaction = CUDA.dot(buffer_state.delta_primal_product, buffer_state.delta_dual) 
    interaction = abs(primal_dual_interaction) 

    norm_delta_primal = CUDA.norm(buffer_state.delta_primal)
    norm_delta_dual = CUDA.norm(buffer_state.delta_dual)

    movement = 0.5 * solver_state.primal_weight * norm_delta_primal^2 + (0.5 / solver_state.primal_weight) * norm_delta_dual^2

    return interaction, movement
end

"""
Take PDHG step with AdaptiveStepsize
"""
function take_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal,
            buffer_state.delta_primal_product,
        )

        compute_next_dual_solution!(
            problem,
            solver_state.current_dual_solution,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal_product,
            solver_state.current_primal_product,
            buffer_state.delta_dual,
        )


        interaction, movement = compute_interaction_and_movement(
            solver_state,
            problem,
            buffer_state,
        )

        solver_state.cumulative_kkt_passes += 1

        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                # The algorithm will terminate at the beginning of the next iteration
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        if step_size <= step_size_limit
            update_solution_in_solver_state!(
                problem,
                solver_state, 
                buffer_state,
            )
            done = true
        end


        first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(step_params.reduction_exponent)) * step_size_limit

        second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(step_params.growth_exponent)) * step_size

        step_size = min(first_term, second_term)
        
    end  
    solver_state.step_size = step_size
end

"""
Take PDHG step with ConstantStepsize
"""
function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal,
        buffer_state.delta_primal_product,
    )
    

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal_product,
        solver_state.current_primal_product,
        buffer_state.delta_dual,
    )

    solver_state.cumulative_kkt_passes += 1

    update_solution_in_solver_state!(
        problem,
        solver_state, 
        buffer_state,
    )
end

function osgm_update!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    osgm_solver_state::OsgmSolverState,
    osgm_stepsize::Float64,
    use_null_step::Bool,
)
    # OSGM block-end hook. Runs every m accepted PDHG steps.

    snap_tol = 32 * eps(Float64)

    # s = z_blk - z_end  (block displacement)
    osgm_solver_state.delta_block_primal .=
        osgm_solver_state.block_primal_solution .- solver_state.current_primal_solution
    osgm_solver_state.delta_block_dual .=
        osgm_solver_state.block_dual_solution .- solver_state.current_dual_solution

    stable_block_update!(
        osgm_solver_state.candidate_primal_solution,
        osgm_solver_state.block_primal_solution,
        solver_state.current_primal_solution,
        osgm_solver_state.primal_preconditioner,
        osgm_solver_state.delta_block_primal;
        snap_tol,
    )

    stable_block_update!(
        osgm_solver_state.candidate_dual_solution,
        osgm_solver_state.block_dual_solution,
        solver_state.current_dual_solution,
        osgm_solver_state.dual_preconditioner,
        osgm_solver_state.delta_block_dual;
        snap_tol,
    )

    # Update A^T y_cand incrementally from A^T y_cur to avoid a full recomputation.
    osgm_solver_state.tmp_dual .=
        osgm_solver_state.candidate_dual_solution .-
        solver_state.current_dual_solution
    osgm_dual_matvec!(
        problem,
        osgm_solver_state.tmp_dual,
        osgm_solver_state.tmp_primal,  # A^T (y_cand - y_cur)
        solver_state,
    )
    # Reconstruct A^T y_cand from the current product plus the delta product.
    osgm_solver_state.candidate_dual_product .=
        solver_state.current_dual_product .+ osgm_solver_state.tmp_primal

    # d = ||r(z_blk)||_M^2  (denominator for OGD update)
    reference_pdhg_residual!(
        problem,
        solver_state,
        osgm_solver_state,
        osgm_solver_state.block_primal_solution,
        osgm_solver_state.block_dual_solution,
        osgm_solver_state.block_dual_product,
    )
    block_residual_norm_sq = residual_metric_norm_squared(
        problem,
        solver_state,
        osgm_solver_state,
    )

    # r(z_osgm), D_x, D_y at candidate; then v = M r
    reference_pdhg_residual!(
        problem,
        solver_state,
        osgm_solver_state,
        osgm_solver_state.candidate_primal_solution,
        osgm_solver_state.candidate_dual_solution,
        osgm_solver_state.candidate_dual_product,
    )
    apply_metric_M!(problem, solver_state, osgm_solver_state)

    # ||r(z_osgm)||_M^2 = r^T M r
    # Must be computed before compute_osgm_gradient! overwrites scratch.
    candidate_residual_norm_sq =
        CUDA.dot(
            osgm_solver_state.residual_primal,
            osgm_solver_state.metric_primal,
        ) +
        CUDA.dot(
            osgm_solver_state.residual_dual,
            osgm_solver_state.metric_dual,
        )

    compute_osgm_gradient!(problem, solver_state, osgm_solver_state)

    gradient_norm_sq =
        CUDA.dot(
            osgm_solver_state.gradient_primal,
            osgm_solver_state.gradient_primal,
        ) +
        CUDA.dot(
            osgm_solver_state.gradient_dual,
            osgm_solver_state.gradient_dual,
        )

    # OGD update with clipping to keep the preconditioners well-scaled.
    if block_residual_norm_sq > eps() &&
        isfinite(block_residual_norm_sq) &&
        isfinite(candidate_residual_norm_sq) &&
        isfinite(gradient_norm_sq)

        step_scale = 2.0 * osgm_stepsize / block_residual_norm_sq

        osgm_solver_state.primal_preconditioner .= clamp.(
            osgm_solver_state.primal_preconditioner .+
            step_scale .* (
                osgm_solver_state.gradient_primal .* osgm_solver_state.delta_block_primal
            ),
            1.e-10, 10.,
        )
        osgm_solver_state.dual_preconditioner .= clamp.(
            osgm_solver_state.dual_preconditioner .+
            step_scale .* (
                osgm_solver_state.gradient_dual .* osgm_solver_state.delta_block_dual
            ),
            1.e-10, 10.,
        )
    end

    # Null-step: accept z_osgm only when ||r(z_osgm)||_M <= ||r(z_blk)||_M.
    # When use_null_step=false, always accept.
    rtol = 1e-12
    atol = 1e-14
    tol = atol + rtol * max(block_residual_norm_sq, candidate_residual_norm_sq)

    if !use_null_step || candidate_residual_norm_sq <= block_residual_norm_sq + tol
        # Update A x_cand incrementally from A x_cur to avoid a full recomputation.
        osgm_solver_state.tmp_primal .=
            osgm_solver_state.candidate_primal_solution .-
            solver_state.current_primal_solution
        osgm_primal_matvec!(
            problem,
            osgm_solver_state.tmp_primal,
            osgm_solver_state.tmp_dual,  # A (x_cand - x_cur)
            solver_state,
        )
        # Reconstruct A x_cand from the current product plus the delta product.
        osgm_solver_state.candidate_primal_product .=
            solver_state.current_primal_product .+ osgm_solver_state.tmp_dual

        solver_state.current_primal_product .=
            osgm_solver_state.candidate_primal_product
        solver_state.current_dual_product .=
            osgm_solver_state.candidate_dual_product
        
        solver_state.current_primal_solution .=
            osgm_solver_state.candidate_primal_solution
        solver_state.current_dual_solution .=
            osgm_solver_state.candidate_dual_solution
    end
    # else: rejected null-step, keep z_end unchanged

    # Start next block from the accepted iterate (z_osgm or z_end)
    sync_osgm_block_start!(solver_state, osgm_solver_state)
end

@inline function stable_block_update!(
    candidate,
    block,
    current,
    preconditioner,
    delta_block;
    snap_tol = 32 * eps(Float64),
)
    @. candidate = ifelse(
        preconditioner <= snap_tol,
        block,
        ifelse(
            abs(1.0 - preconditioner) <= snap_tol,
            current,
            ifelse(
                abs(preconditioner) <= abs(1.0 - preconditioner),
                muladd(-preconditioner, delta_block, block),          # b - p*d
                muladd(1.0 - preconditioner, delta_block, current),  # c + (1-p)*d
            ),
        ),
    )
    return
end

"""
Main algorithm: given parameters and LP problem, return solutions
"""
function optimize(
    params::PdhgParameters,
    osgm_params::OsgmParams,
    original_problem::QuadraticProgrammingProblem,
)
    validate(original_problem)
    qp_cache = cached_quadratic_program_info(original_problem)

    start_rescaling_time = time()
    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )
    rescaling_time = time() - start_rescaling_time
    if params.verbosity >= 1
        Printf.@printf(
            "Preconditioning Time (seconds): %.2e\n",
            rescaling_time,
        )
    end

    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end
    if osgm_params.osgm_blocksize <= 0
        error("osgm_blocksize must be positive")
    end

    # transfer from cpu to gpu
    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    d_problem = d_scaled_problem.scaled_qp
    buffer_lp = qp_cpu_to_gpu(original_problem)

    # initialization
    solver_state = CuPdhgSolverState(
        CUDA.zeros(Float64, primal_size),    # current_primal_solution
        CUDA.zeros(Float64, dual_size),      # current_dual_solution
        CUDA.zeros(Float64, dual_size),      # current_primal_product
        CUDA.zeros(Float64, primal_size),    # current_dual_product
        initialize_solution_weighted_average(primal_size, dual_size),
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        nothing,
        nothing,
    )

    buffer_state = CuBufferState(
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # delta_primal_product
    )

    buffer_avg = CuBufferAvgState(
        CUDA.zeros(Float64, primal_size),      # avg_primal_solution
        CUDA.zeros(Float64, dual_size),        # avg_dual_solution
        CUDA.zeros(Float64, dual_size),        # avg_primal_product
        CUDA.zeros(Float64, primal_size),      # avg_primal_gradient
    )

    buffer_original = BufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
    )

    buffer_kkt = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_kkt_infeas = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
    buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product

    # stepsize
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        solver_state.cumulative_kkt_passes += 0.5
        solver_state.step_size = 1.0 / norm(scaled_problem.scaled_qp.constraint_matrix, Inf)
    else
        solver_state.step_size, number_of_power_iterations =
            estimate_reference_step_size(scaled_problem.scaled_qp.constraint_matrix)
        solver_state.cumulative_kkt_passes += number_of_power_iterations
    end

    reference_step_size = solver_state.step_size
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        reference_step_size, number_of_power_iterations =
            estimate_reference_step_size(scaled_problem.scaled_qp.constraint_matrix)
        solver_state.cumulative_kkt_passes += number_of_power_iterations
    end

    osgm_solver_state = OsgmSolverState(
        CUDA.ones(Float64, primal_size),      # primal_preconditioner
        CUDA.ones(Float64, dual_size),        # dual_preconditioner
        CUDA.zeros(Float64, primal_size),     # block_primal_solution
        CUDA.zeros(Float64, dual_size),       # block_dual_solution
        CUDA.zeros(Float64, dual_size),       # block_primal_product
        CUDA.zeros(Float64, primal_size),     # block_dual_product
        reference_step_size,                  # tau_ref
        reference_step_size,                  # sigma_ref
        0,                                    # accepted_steps_in_block
        CUDA.zeros(Float64, primal_size),     # delta_block_primal
        CUDA.zeros(Float64, dual_size),       # delta_block_dual
        CUDA.zeros(Float64, primal_size),     # candidate_primal_solution
        CUDA.zeros(Float64, dual_size),       # candidate_dual_solution
        CUDA.zeros(Float64, dual_size),       # candidate_primal_product
        CUDA.zeros(Float64, primal_size),     # candidate_dual_product
        CUDA.zeros(Float64, primal_size),     # reference_primal_solution
        CUDA.zeros(Float64, dual_size),       # reference_dual_solution
        CUDA.zeros(Float64, primal_size),     # residual_primal
        CUDA.zeros(Float64, dual_size),       # residual_dual
        CUDA.zeros(Float64, primal_size),     # primal_mask
        CUDA.zeros(Float64, dual_size),       # dual_mask
        CUDA.zeros(Float64, primal_size),     # metric_primal
        CUDA.zeros(Float64, dual_size),       # metric_dual
        CUDA.zeros(Float64, primal_size),     # gradient_primal
        CUDA.zeros(Float64, dual_size),       # gradient_dual
        CUDA.zeros(Float64, primal_size),     # tmp_primal
        CUDA.zeros(Float64, dual_size),       # tmp_dual
    )

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            d_scaled_problem.scaled_qp,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        buffer_primal_gradient,
    )
    sync_osgm_block_start!(solver_state, osgm_solver_state)

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading(params.verbosity)

    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.sum_primal_solutions_count == 0 || solver_state.solution_weighted_avg.sum_dual_solutions_count == 0
                buffer_avg.avg_primal_solution .= solver_state.current_primal_solution
                buffer_avg.avg_dual_solution .= solver_state.current_dual_solution
                buffer_avg.avg_primal_product .= solver_state.current_primal_product
                buffer_avg.avg_primal_gradient .= buffer_primal_gradient
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            ### KKT ###
            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
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
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            if iteration < 10 && (termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE || termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
                termination_reason = false
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats, params.verbosity)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                
                # GPU to CPU
                avg_primal_solution = zeros(primal_size)
                avg_dual_solution = zeros(dual_size)
                gpu_to_cpu!(
                    buffer_avg.avg_primal_solution,
                    buffer_avg.avg_dual_solution,
                    avg_primal_solution,
                    avg_dual_solution,
                )

                pdhg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    iteration,
                    termination_reason,
                    current_iteration_stats,
                )

                return unscaled_saddle_point_output(
                    scaled_problem,
                    avg_primal_solution,
                    avg_dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                )
            end

            buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product


            current_iteration_stats.restart_used = run_restart_scheme(
                d_scaled_problem.scaled_qp,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
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

            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
                solver_state.ratio_step_sizes = 1.0
                sync_osgm_block_start!(solver_state, osgm_solver_state)
            end
        end

        time_spent_doing_basic_algorithm_checkpoint = time()
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                # problem,
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end
        
        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        # Count accepted PDHG steps toward the current block (spec §8).
        if !solver_state.numerical_error
            osgm_solver_state.accepted_steps_in_block += 1
        end

        # # Block-end hook: run OSGM update every m accepted steps (spec §8).
        if osgm_solver_state.accepted_steps_in_block >= osgm_params.osgm_blocksize
            osgm_update!(
                d_problem, solver_state, buffer_state, osgm_solver_state,
                osgm_params.osgm_stepsize, osgm_params.use_null_step,
            )
        end

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end

function optimize(
    params::PdhgParameters,
    original_problem::QuadraticProgrammingProblem,
)
    return optimize(params, OsgmPdhgParameters(0.0, 10, false), original_problem)
end
