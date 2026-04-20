import ArgParse

import cuPDLP

include(joinpath(@__DIR__, "solve.jl"))

const SUPPORTED_INSTANCE_SUFFIXES = (
    ".mps",
    ".mps.gz",
    ".qps",
    ".qps.gz",
)

function is_supported_instance_path(path::String)::Bool
    lower_path = lowercase(path)
    return any(suffix -> endswith(lower_path, suffix), SUPPORTED_INSTANCE_SUFFIXES)
end

function collect_instance_paths(dataset_directory::String; recursive::Bool)::Vector{String}
    instance_paths = String[]

    if recursive
        for (root, _dirs, files) in walkdir(dataset_directory)
            for file in files
                path = joinpath(root, file)
                if is_supported_instance_path(path)
                    push!(instance_paths, path)
                end
            end
        end
    else
        for file in readdir(dataset_directory)
            path = joinpath(dataset_directory, file)
            if isfile(path) && is_supported_instance_path(path)
                push!(instance_paths, path)
            end
        end
    end

    sort!(instance_paths)
    return instance_paths
end

function output_dir_for_instance(
    instance_path::String,
    dataset_directory::String,
    output_directory::String,
)::String
    relative_dir = relpath(dirname(instance_path), dataset_directory)
    if relative_dir == "." || isempty(relative_dir)
        return output_directory
    end
    return joinpath(output_directory, relative_dir)
end

function summary_output_path_for_instance(output_dir::String, instance_path::String)::String
    instance_name = instance_name_from_path(instance_path)
    return joinpath(output_dir, instance_name * "_summary.json")
end

function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--dataset_directory"
        help = "Directory containing LP/QP instances (e.g. .mps(.gz), .qps(.gz))."
        arg_type = String
        required = true

        "--output_directory"
        help = "Base directory for output files (mirrors dataset subdirectories)."
        arg_type = String
        required = true

        "--recursive"
        help = "Whether to search dataset_directory recursively."
        arg_type = Bool
        default = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-4

        "--time_sec_limit"
        help = "Time limit per instance."
        arg_type = Float64
        default = 3600.0

        "--osgm_stepsize"
        help = "OSGM hypergradient stepsize η (0.0 makes OSGM a no-op and uses plain PDHG)."
        arg_type = Float64
        default = 0.0

        "--osgm_block_size"
        help = "OSGM block size m (number of inner PDHG steps per outer OSGM step)."
        arg_type = Int
        default = 64

        "--skip_existing"
        help = "Skip instances whose *_summary.json already exists."
        action = :store_true

        "--limit"
        help = "Solve at most this many instances (0 means no limit)."
        arg_type = Int
        default = 0

        "--dry_run"
        help = "Print discovered instances and exit without solving."
        action = :store_true

        "--fail_fast"
        help = "Stop immediately if any instance fails."
        action = :store_true

        "--skip_warmup"
        help = "Skip warm-up run (useful if you already precompiled / warmed up)."
        action = :store_true

        "--warmup_instance_path"
        help = "Optional instance path to use for warm-up (defaults to smallest discovered file)."
        arg_type = String
        default = ""
    end

    return ArgParse.parse_args(arg_parse)
end

function build_solver_params(tolerance::Float64, time_sec_limit::Float64)
    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    return cuPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        2,
        true,
        64,
        termination_params,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3, 0.6),
    )
end

function warm_up_once(instance_path::String)
    lp = cuPDLP.qps_reader_to_standard_form(instance_path)

    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp)
    redirect_stdout(oldstd)
end

function choose_default_warmup_instance(instance_paths::Vector{String})::String
    best_path = instance_paths[1]
    best_size = filesize(best_path)

    for path in Iterators.drop(instance_paths, 1)
        size = filesize(path)
        if size < best_size
            best_path = path
            best_size = size
        end
    end

    return best_path
end

function main()
    parsed_args = parse_command_line()

    dataset_directory = abspath(parsed_args["dataset_directory"])
    output_directory = abspath(parsed_args["output_directory"])
    recursive = parsed_args["recursive"]

    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    osgm_stepsize = parsed_args["osgm_stepsize"]
    osgm_block_size = parsed_args["osgm_block_size"]

    skip_existing = parsed_args["skip_existing"]
    dry_run = parsed_args["dry_run"]
    limit = parsed_args["limit"]
    fail_fast = parsed_args["fail_fast"]
    skip_warmup = parsed_args["skip_warmup"]
    warmup_instance_path = parsed_args["warmup_instance_path"]

    if !isdir(dataset_directory)
        error("dataset_directory does not exist or is not a directory: ", dataset_directory)
    end

    instance_paths = collect_instance_paths(dataset_directory; recursive = recursive)
    if limit > 0
        instance_paths = instance_paths[1:min(limit, length(instance_paths))]
    end

    println("Found ", length(instance_paths), " instance(s) under ", dataset_directory)

    if dry_run
        try
            for path in instance_paths
                println(path)
            end
        catch err
            if err isa Base.IOError && err.code == Base.UV_EPIPE
                return
            end
            rethrow()
        end
        return
    end

    if isempty(instance_paths)
        error("No supported instances found under: ", dataset_directory)
    end

    if !skip_warmup
        if isempty(warmup_instance_path)
            warmup_instance_path = choose_default_warmup_instance(instance_paths)
        end
        println("Warm-up instance: ", warmup_instance_path)
        warm_up_once(warmup_instance_path)
    end

    params = build_solver_params(tolerance, time_sec_limit)
    osgm_params = cuPDLP.OsgmPdhgParameters(osgm_stepsize, osgm_block_size)

    start_time = time()
    solved_count = 0
    skipped_count = 0
    failed_count = 0

    for (idx, instance_path) in enumerate(instance_paths)
        instance_output_dir =
            output_dir_for_instance(instance_path, dataset_directory, output_directory)
        summary_output_path =
            summary_output_path_for_instance(instance_output_dir, instance_path)

        if skip_existing && isfile(summary_output_path)
            println("[", idx, "/", length(instance_paths), "] skip: ", instance_path)
            skipped_count += 1
            continue
        end

        println("[", idx, "/", length(instance_paths), "] solve: ", instance_path)
        try
            solve_instance_and_output(params, osgm_params, instance_output_dir, instance_path)
            solved_count += 1
        catch err
            failed_count += 1
            mkpath(instance_output_dir)

            error_output_path =
                joinpath(instance_output_dir, instance_name_from_path(instance_path) * "_error.txt")
            open(error_output_path, "w") do io
                println(io, "Instance: ", instance_path)
                println(io)
                showerror(io, err)
                println(io)
                println(io)
                Base.show_backtrace(io, catch_backtrace())
            end

            println("  failed: wrote ", error_output_path)
            if fail_fast
                rethrow()
            end
        end
    end

    elapsed = time() - start_time
    println(
        "Done. solved=",
        solved_count,
        " skipped=",
        skipped_count,
        " failed=",
        failed_count,
        " elapsed_sec=",
        round(elapsed; digits = 2),
    )
end

main()
