---
name: cupdhg:cuda
description: Use when writing or debugging CUDA kernels in cuPDLP.jl, using cuSPARSE sparse matrix operations, allocating CuArrays, translating CPU Julia code to GPU, or understanding GPU-specific patterns used in this repo. Do NOT use for general Julia idioms — the Julia MCP covers those.
---

# cuPDLP.jl Julia/CUDA Idioms

## Kernel Launch Pattern

```julia
# Canonical pattern used throughout this repo:
NumBlock = ceil(Int64, n / ThreadPerBlock)   # NOT cld() — use ceil(Int64, ...)
CUDA.@sync @cuda threads=ThreadPerBlock blocks=NumBlock my_kernel!(arg1, arg2, n)
```

`ThreadPerBlock = 128` (defined in `src/cuPDLP.jl`). Always use `CUDA.@sync` before the kernel — it ensures the kernel completes before the next CPU operation.

## Kernel Signature Convention

```julia
function my_kernel!(
    vec_a::CuDeviceVector{Float64},   # GPU arrays use CuDeviceVector inside kernels
    vec_b::CuDeviceVector{Float64},
    n::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= n
        @inbounds begin
            vec_a[tx] = vec_b[tx] * 2.0   # all logic inside @inbounds
        end
    end
    return                                 # explicit return, no value
end
```

Outside kernels: `CuVector{Float64}`. Inside kernels: `CuDeviceVector{Float64}`. CUDA.jl converts automatically on `@cuda` launch.

## cuSPARSE Sparse Matrix-Vector Product

```julia
# Compute: out = alpha * A * x + beta * out
# Used for A*Δx (constraint matrix, 'N' = no transpose):
CUDA.CUSPARSE.mv!(
    'N',                                      # 'N' = A*x, 'T' = Aᵀ*x
    1,                                        # alpha
    problem.constraint_matrix,               # CuSparseMatrix (CSR format)
    delta_primal,                             # x: CuVector{Float64}
    0,                                        # beta
    delta_primal_product,                     # out: CuVector{Float64}
    'O',                                      # index base: 'O' = 0-based (CUDA), 'I' = 1-based
    CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2,    # algorithm selector
)
```

The constraint matrix is stored in CSR format. `constraint_matrix` is `CuLinearProgrammingProblem.constraint_matrix`. For `Aᵀy` use `'T'`.

## CuArray Allocation

```julia
# Pre-allocate in state struct constructor — never inside iteration loops:
mutable struct CuMyState
    my_vec::CuVector{Float64}
end

function init_my_state(n::Int64)
    CuMyState(CUDA.zeros(Float64, n))    # zeros
    # or: CUDA.ones(Float64, n)
    # or: similar(existing_cu_vec)        # same size/type, uninitialized
    # or: CuArray(cpu_vec)                # copy from CPU
end
```

**Never allocate inside `take_step!` or kernel wrapper functions** — allocations in hot loops cause GC pressure and tank performance. Pre-allocate everything in the state struct.

## Dot Products and Norms on GPU

```julia
# Correct: stays on GPU, returns scalar
val = CUDA.dot(vec_a, vec_b)    # not LinearAlgebra.dot
norm_val = CUDA.norm(vec_a)      # not LinearAlgebra.norm

# Wrong: triggers scalar indexing (will error or be slow):
# val = sum(vec_a .* vec_b)      ← allocates temp CuArray
```

## Elementwise Operations

```julia
# Use broadcasting — executes as GPU kernel:
vec_a .= vec_b .* scalar          # in-place, no allocation
vec_a .= max.(vec_b, 0.0)         # elementwise max

# For fused operations, write an explicit kernel (avoids multiple passes)
```

## Active-Set Mask Pattern

Pattern from `update_primal_hyper_state_kernel!` (line 488, `osgm_pdhg.jl`):

```julia
function compute_mask_kernel!(
    x::CuDeviceVector{Float64},
    lb::CuDeviceVector{Float64},
    ub::CuDeviceVector{Float64},
    mask::CuDeviceVector{Float64},
    n::Int64,
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= n
        @inbounds begin
            # 1.0 if strictly interior to box, 0.0 if at a bound
            mask[tx] = (x[tx] > lb[tx] && x[tx] < ub[tx]) ? 1.0 : 0.0
        end
    end
    return
end
```

Then multiply downstream vectors by the mask elementwise rather than branching inside subsequent kernels.

## pdhg_step_pure! Pattern

When you need to evaluate `T(z)` without mutating solver state (e.g., for computing block residuals in OSGM):

```julia
# pdhg_step_pure! writes into caller-supplied output buffers
# It does NOT touch solver_state
pdhg_step_pure!(
    problem,
    solver_state,          # read-only in this call
    output_primal,         # CuVector: where to write T(z).x
    output_dual,           # CuVector: where to write T(z).y
    step_size,
    primal_weight,
    extrapolation_coefficient,
)
```

Use this instead of `take_step!` whenever you need the output of `T` without advancing the solver state.

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| `CUDA.@allowscalar` in hot path | Rewrite as kernel — scalar indexing serializes to CPU |
| `synchronize()` missing before timing | Add `CUDA.synchronize()` before `time()` calls |
| `Float32` vs `Float64` mismatch | All state arrays are `Float64` — check `eltype` when creating new CuArrays |
| Block count with `div` | Use `ceil(Int64, n/ThreadPerBlock)` — `div` drops the last partial block |
| Allocating in `take_step!` | Pre-allocate in state struct; pass buffers as arguments |
| `norm` on `CuVector` giving wrong result | Use `CUDA.norm`, not `LinearAlgebra.norm` — the latter may trigger scalar indexing |
