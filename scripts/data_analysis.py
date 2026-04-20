import pandas as pd
import numpy as np

def shifted_geometric_mean(data, shift=1):
    shifted_data = data + shift
    return np.exp(np.mean(np.log(shifted_data)))

def compute_shifted_geometric_mean(df, selected_columns, shift=1):
    selected_data = df[selected_columns]
    return selected_data.apply(lambda x: shifted_geometric_mean(x, shift))

def compute_mean(df, selected_columns):
    return df[selected_columns].mean()

def count_instances(df, column, reference_column):
    total_length = len(df)
    count_better = (df[column] < df[reference_column]).sum()
    count_equal = (df[column] == df[reference_column]).sum()
    count_worse = (df[column] > df[reference_column]).sum()
    return count_better, count_equal, count_worse

# file_path = 'output/table/netlib/merged_time_3600.0_tol_0.0001.csv'
file_path = "output/table/MIPLIB/merged_time_3600.0_tol_0.0001.csv"

shift = 10
df = pd.read_csv(file_path)

termination_columns = [col for col in df.columns if col.startswith('termination_string')]
df_filtered = df[df[termination_columns].apply(lambda x: all(x == 'OPTIMAL'), axis=1)]

iteration_cols = [col for col in df.columns if col.startswith('iteration_count')]
solve_time_cols = [col for col in df.columns if col.startswith('solve_time_sec')]

df_filtered = df_filtered[df_filtered[iteration_cols].apply(lambda x: all(x >= 10000), axis=1)]

results = []
for i in range(len(iteration_cols)):
    iteration_col = iteration_cols[i]
    solve_time_col = solve_time_cols[i]
    termination_col = termination_columns[i]
    count_opt = len(df[df[termination_col]=='OPTIMAL'])
    iter_SGM = round(compute_shifted_geometric_mean(df_filtered, [iteration_col], shift).values[0], 2)
    iter_mean = round(compute_mean(df_filtered, [iteration_col]).values[0], 2)
    time_SGM = round(compute_shifted_geometric_mean(df_filtered, [solve_time_col], 0).values[0], 2)
    time_mean = round(compute_mean(df_filtered, [solve_time_col]).values[0], 2)
    count_better, count_equal, count_worse = count_instances(df_filtered, iteration_col, 'iteration_count_PDLP')
    method = iteration_col.split('_')[-1]
    results.append({
        'Method': method,
        'OPT Count': count_opt,
        'Iteration SGM': iter_SGM,
        'Iteration Mean': iter_mean,
        'Time SGM': time_SGM,
        'Time Mean': time_mean,
        'Better': count_better,
        'Equal': count_equal,
        'Worse': count_worse
    })

import os
results_df = pd.DataFrame(results)
output_file = file_path.replace('table', 'stats')
output_dir = '/'.join(output_file.split('/')[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
results_df.to_csv(output_file, index=False)
print(results_df.to_string(index=False))
