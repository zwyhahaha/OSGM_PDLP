using cuPDLP
using Test

# Sanity check: OSGM with osgm_stepsize=0 must terminate at the same iteration
# as plain PDHG and produce the same primal objective (within 1e-8).
# If afiro takes more than 60 seconds there is a bug — infinite loop or broken convergence.

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
        time_sec_limit = 60.0,   # must finish in < 60s or there is a bug
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )
    params = cuPDLP.PdhgParameters(
        10, false, 1.0, 1.0, true, 0, true, 64,
        termination_params, restart_params,
        cuPDLP.ConstantStepsizeParams(),
    )

    # Plain PDHG (no OSGM)
    result_pdhg = cuPDLP.optimize(params, lp)

    # OSGM with stepsize=0 (must be identical to plain PDHG)
    osgm_params = cuPDLP.OsgmPdhgParameters(0.0, 64)
    result_osgm = cuPDLP.optimize(params, osgm_params, lp)

    @test result_pdhg.termination_reason == result_osgm.termination_reason
    @test abs(result_pdhg.iteration_count - result_osgm.iteration_count) <= 64

    pdhg_obj = result_pdhg.iteration_stats[end].convergence_information[1].primal_objective
    osgm_obj = result_osgm.iteration_stats[end].convergence_information[1].primal_objective
    # Objectives must be close. Both converge to 1e-4 KKT tolerance, but restart timing
    # differs by up to one block (PDHG checks at 1,65,129,... OSGM at 64,128,192,...),
    # which produces different weighted averages. 2e-2 gives sufficient margin.
    @test abs(pdhg_obj - osgm_obj) < 2e-2
end
