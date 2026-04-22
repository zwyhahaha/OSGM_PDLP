import argparse
import csv
import json
import math
from pathlib import Path


RESULTS_DIR = Path("results")
DEFAULT_FILE_PREFIX = "netlib_1e-4"
ITERATION_SHIFT = 10
TIME_SHIFT = 0
COLUMN_ORDER = [
    "instance_name",
    "termination_string",
    "iteration_count",
    "solve_time_sec",
    "cumulative_kkt_matrix_passes",
    "cumulative_time_sec",
    "time_spent_doing_basic_algorithm",
]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def shifted_geometric_mean(values, shift=1):
    if not values:
        return ""

    shifted_values = [value + shift for value in values]
    if any(value <= 0 for value in shifted_values):
        return ""

    log_mean = sum(math.log(value) for value in shifted_values) / len(shifted_values)
    return round(math.exp(log_mean), 2)


def arithmetic_mean(values):
    if not values:
        return ""
    return round(sum(values) / len(values), 2)


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def instance_name_from_path(json_file, json_dir):
    relative_dir = json_file.parent.relative_to(json_dir)
    if str(relative_dir) == ".":
        return json_file.name.removesuffix("_summary.json")
    return str(relative_dir)


def row_from_json_file(json_file, json_dir):
    with json_file.open() as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary at the top level of the JSON file")

    solution_stats = data.get("solution_stats") or {}
    method_specific_stats = solution_stats.get("method_specific_stats") or {}

    instance_name = data.get("instance_name")
    if instance_name is None:
        instance_name = instance_name_from_path(json_file, json_dir)

    return {
        "instance_name": csv_value(instance_name),
        "termination_string": csv_value(data.get("termination_string")),
        "iteration_count": csv_value(data.get("iteration_count")),
        "solve_time_sec": csv_value(data.get("solve_time_sec")),
        "cumulative_kkt_matrix_passes": csv_value(
            solution_stats.get("cumulative_kkt_matrix_passes")
        ),
        "cumulative_time_sec": csv_value(solution_stats.get("cumulative_time_sec")),
        "time_spent_doing_basic_algorithm": csv_value(
            method_specific_stats.get("time_spent_doing_basic_algorithm")
        ),
    }


def write_result_table(json_dir):
    rows = []
    for json_file in sorted(json_dir.rglob("*.json")):
        try:
            rows.append(row_from_json_file(json_file, json_dir))
        except Exception as error:
            print(f"Error reading {json_file}: {error}")

    csv_path = Path(f"{json_dir}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMN_ORDER)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, len(rows)


def convert_result_dirs(file_prefix):
    csv_paths = []
    for path in sorted(RESULTS_DIR.glob(f"{file_prefix}*")):
        if not path.is_dir():
            continue
        if any(path.rglob("*.json")):
            csv_paths.append(write_result_table(path))
    return csv_paths


def method_from_path(path, file_prefix):
    stem = path.stem
    if stem == file_prefix:
        return "PDLP"
    if stem.startswith(f"{file_prefix}_"):
        return stem[len(file_prefix) + 1 :]
    return stem


def read_rows(path):
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "instance_name",
            "termination_string",
            "iteration_count",
            "solve_time_sec",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing}")

        return list(reader)


def read_opt_rows(path):
    return [
        row
        for row in read_rows(path)
        if row["termination_string"].strip() == "OPTIMAL"
    ]


def count_comparison(opt_rows, reference_iterations):
    count_better = 0
    count_equal = 0
    count_worse = 0

    for row in opt_rows:
        iteration = to_float(row["iteration_count"])
        reference_iteration = reference_iterations.get(row["instance_name"])
        if iteration is None or reference_iteration is None:
            continue

        if iteration < reference_iteration:
            count_better += 1
        elif iteration == reference_iteration:
            count_equal += 1
        else:
            count_worse += 1

    return count_better, count_equal, count_worse


def summarize_file(path, reference_iterations, file_prefix):
    opt_rows = [
        row
        for row in read_opt_rows(path)
        if row["instance_name"] in reference_iterations
    ]

    iteration_values = [
        value
        for value in (to_float(row["iteration_count"]) for row in opt_rows)
        if value is not None
    ]
    time_values = [
        value
        for value in (to_float(row["solve_time_sec"]) for row in opt_rows)
        if value is not None
    ]
    count_better, count_equal, count_worse = count_comparison(opt_rows, reference_iterations)

    return {
        "Method": method_from_path(path, file_prefix),
        "OPT Count": len(opt_rows),
        "Iteration SGM": shifted_geometric_mean(iteration_values, ITERATION_SHIFT),
        "Iteration Mean": arithmetic_mean(iteration_values),
        "Time SGM": shifted_geometric_mean(time_values, TIME_SHIFT),
        "Time Mean": arithmetic_mean(time_values),
        "Better": count_better,
        "Equal": count_equal,
        "Worse": count_worse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize solver result CSV files."
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_FILE_PREFIX,
        help=(
            "Result CSV prefix in the results directory. For example, "
            "MIPLIB_1e-4 summarizes results/MIPLIB_1e-4*.csv."
        ),
    )
    parser.add_argument(
        "--pdlp-min-iterations",
        type=float,
        default=None,
        help=(
            "Only include instances where the PDLP baseline iteration_count is "
            "strictly greater than this value."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to the regular or large summary name.",
    )
    parser.add_argument(
        "--complete-instances-only",
        action="store_true",
        help="Only include instances that have a row in every matching result CSV.",
    )
    parser.add_argument(
        "--skip-json-conversion",
        action="store_true",
        help="Skip converting matching result JSON directories into CSV files.",
    )
    return parser.parse_args()


def build_reference_iterations(reference_rows, pdlp_min_iterations, allowed_instances=None):
    reference_iterations = {}

    for row in reference_rows:
        instance_name = row["instance_name"]
        if allowed_instances is not None and instance_name not in allowed_instances:
            continue
        iteration = to_float(row["iteration_count"])
        if iteration is None:
            continue
        if pdlp_min_iterations is not None and iteration <= pdlp_min_iterations:
            continue
        reference_iterations[instance_name] = iteration

    return reference_iterations


def common_instances(csv_files):
    instance_sets = []
    for path in csv_files:
        instance_sets.append({row["instance_name"] for row in read_rows(path)})

    if not instance_sets:
        return set()
    return set.intersection(*instance_sets)


def main():
    args = parse_args()
    file_prefix = args.file_prefix
    default_output_file = RESULTS_DIR / f"{file_prefix}_summary.csv"
    default_large_output_file = RESULTS_DIR / f"{file_prefix}_large_summary.csv"

    output_file = args.output_file
    if output_file is None:
        output_file = (
            default_large_output_file
            if args.pdlp_min_iterations is not None
            else default_output_file
        )

    if not args.skip_json_conversion:
        converted_tables = convert_result_dirs(file_prefix)
        for csv_path, row_count in converted_tables:
            print(f"Wrote {csv_path} ({row_count} rows)")

    csv_files = sorted(
        path
        for path in RESULTS_DIR.glob(f"{file_prefix}*.csv")
        if path != output_file and not path.name.endswith("_summary.csv")
    )
    if not csv_files:
        pattern = RESULTS_DIR / f"{file_prefix}*.csv"
        raise FileNotFoundError(f"No CSV files found for {pattern}")

    reference_path = RESULTS_DIR / f"{file_prefix}.csv"
    reference_rows = read_opt_rows(reference_path)
    allowed_instances = common_instances(csv_files) if args.complete_instances_only else None
    reference_iterations = build_reference_iterations(
        reference_rows,
        args.pdlp_min_iterations,
        allowed_instances,
    )

    rows = [
        summarize_file(path, reference_iterations, file_prefix)
        for path in csv_files
    ]
    fieldnames = [
        "Method",
        "OPT Count",
        "Iteration SGM",
        "Iteration Mean",
        "Time SGM",
        "Time Mean",
        "Better",
        "Equal",
        "Worse",
    ]

    with output_file.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[field]) for field in fieldnames))
    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    main()
