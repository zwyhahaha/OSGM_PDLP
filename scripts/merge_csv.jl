
using ArgParse
using CSV
using DataFrames

function merge_table_general(df_list,name_list)
    for i in eachindex(df_list)
        result_dict  = Dict(
            "termination_string" => "termination_string_$(name_list[i])",
            "iteration_count" => "iteration_count_$(name_list[i])",
            "solve_time_sec" => "solve_time_sec_$(name_list[i])",
            "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_$(name_list[i])",
            "cumulative_time_sec" => "cumulative_time_sec_$(name_list[i])",
            "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_$(name_list[i])"
        )
        if "learning_rate" in names(df_list[i])
            result_dict["learning_rate"] = "learning_rate_$(name_list[i])"
        end
        rename!(df_list[i], result_dict)
    end
    merged_df = outerjoin(df_list..., on=:instance_name, makeunique=true)
    column_names = names(merged_df)
    selected_columns = filter(name -> startswith(name, "termination_string") || startswith(name, "iteration_count") || startswith(name, "solve_time_sec") || startswith(name, "learning_rate"), column_names)
    selected_columns = sort(selected_columns)
    selected_columns = ["instance_name"; selected_columns]
    merged_df = merged_df[:, selected_columns]
    return merged_df
end

function merge_table(os_df, ada_df, ifos_df)
    rename!(ada_df, Dict(
        "termination_string" => "termination_string_adaPDLP",
        "iteration_count" => "iteration_count_adaPDLP",
        "solve_time_sec" => "solve_time_sec_adaPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_adaPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_adaPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_adaPDLP"
    ))

    rename!(ifos_df, Dict(
        "termination_string" => "termination_string_ifosPDLP",
        "iteration_count" => "iteration_count_ifosPDLP",
        "solve_time_sec" => "solve_time_sec_ifosPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_ifosPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_ifosPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_ifosPDLP",
        "learning_rate" => "learning_rate_ifosPDLP"
    ))

    rename!(os_df, Dict(
        "termination_string" => "termination_string_osPDLP",
        "iteration_count" => "iteration_count_osPDLP",
        "solve_time_sec" => "solve_time_sec_osPDLP",
        "cumulative_kkt_matrix_passes" => "cumulative_kkt_matrix_passes_osPDLP",
        "cumulative_time_sec" => "cumulative_time_sec_osPDLP",
        "time_spent_doing_basic_algorithm" => "time_spent_doing_basic_algorithm_osPDLP",
        "learning_rate" => "learning_rate_osPDLP"
    ))

    df_merged = outerjoin(ada_df, ifos_df, os_df, on=:instance_name)

    column_order = [
        "instance_name",
        "learning_rate_osPDLP", "learning_rate_ifosPDLP",
        "termination_string_adaPDLP", "termination_string_osPDLP", "termination_string_ifosPDLP",
        "iteration_count_adaPDLP", "iteration_count_osPDLP", "iteration_count_ifosPDLP",
        "solve_time_sec_adaPDLP", "solve_time_sec_osPDLP", "solve_time_sec_ifosPDLP",
        "cumulative_kkt_matrix_passes_adaPDLP", "cumulative_kkt_matrix_passes_osPDLP", "cumulative_kkt_matrix_passes_ifosPDLP",
        "cumulative_time_sec_adaPDLP", "cumulative_time_sec_osPDLP", "cumulative_time_sec_ifosPDLP",
        "time_spent_doing_basic_algorithm_adaPDLP", "time_spent_doing_basic_algorithm_osPDLP", "time_spent_doing_basic_algorithm_ifosPDLP"
    ]

    df_merged = df_merged[:, column_order]

    return df_merged

    
end

function merge_os_dfs(experiment::String, dataset::String, time_limit::Float64, tolerance::Float64)
    dir_path = "output/table/$(dataset)"
    pattern  = "$(experiment)_time_$(time_limit)_tol_$(tolerance)"
    files = filter(f -> startswith(f, pattern), readdir(dir_path))

    dfs = DataFrame[]
    for file in files
        lr_match = match(r"lr_([\d\.eE+-]+)\.csv", file)
        lr_val   = (lr_match === nothing) ? missing : parse(Float64, lr_match.captures[1])
        df       = CSV.read(joinpath(dir_path, file), DataFrame; stringtype=String)
        df[!, :learning_rate] = fill(lr_val, nrow(df))
        push!(dfs, df)
    end

    if isempty(dfs)
        return DataFrame()
    end

    merged_df = dfs[length(dfs)]
    for i in 1:length(dfs)-1
        df = dfs[i]
        for row in eachrow(df)
            instance_name = row[:instance_name]
            idx = findfirst(isequal(instance_name), merged_df[!, :instance_name])
            if idx !== nothing
                if row[:iteration_count] < merged_df[idx, :iteration_count]
                    common_cols = intersect(names(merged_df), names(row))
                    for col in common_cols
                        merged_df[idx, col] = coalesce(row[col], merged_df[idx, col])
                    end
                end
            else
                # new_row = DataFrame()
                # for col in names(row)
                #     new_row[!, Symbol(col)] = [coalesce(row[col], missing)]
                # end
                # push!(merged_df, new_row[1, :])
            end
        end
    end
    return merged_df
end


function parse_command_line()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--dataset"
        help = "Path to the dataset"
        arg_type = String

        "--time_sec_limit"
        help = "Time limit for the experiments"
        arg_type = Float64

        "--tolerance"
        help = "Tolerance for the experiments"
        arg_type = Float64
    end
    return ArgParse.parse_args(s)
end

args = parse_command_line()
dataset = args["dataset"]
time_limit = args["time_sec_limit"]
tolerance = args["tolerance"]

ada_file = "output/table/$(dataset)/PDLP_time_$(time_limit)_tol_$(tolerance)_lr_0.0.csv"
ada_df = CSV.read(ada_file, DataFrame)
os_df = merge_os_dfs("wn-f-adaPDLP",dataset, time_limit, tolerance)
osnon_df = merge_os_dfs("won-f-adaPDLP",dataset, time_limit, tolerance)
if20_df = merge_os_dfs("wn-if-adaPDLP",dataset, time_limit, tolerance)
if20non_df = merge_os_dfs("won-if-adaPDLP",dataset, time_limit, tolerance)
df_merged = merge_table_general([ada_df, os_df, osnon_df, if20_df, if20non_df], ["PDLP", "wn-f", "won-f", "wn-if", "won-if"])
CSV.write("output/table/$(dataset)/merged_time_$(time_limit)_tol_$(tolerance).csv", df_merged)