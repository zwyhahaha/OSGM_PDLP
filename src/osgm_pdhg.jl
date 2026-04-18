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
