import pandas as pd
import os

data = [
    {'env': 'Hopper-v4', 'type': 'basic', 'excel': './results/processed_Hopper_basic.xlsx'},
    {'env': 'Hopper-v4', 'type': 'static', 'excel': './results/processed_Hopper_static.xlsx'},
    {'env': 'Hopper-v4', 'type': 'dynamic', 'excel': './results/processed_Hopper_dynamic.xlsx'},
    {'env': 'Walker-v4', 'type': 'basic', 'excel': './results/processed_Walker2d_basic.xlsx'},
    {'env': 'Walker-v4', 'type': 'static', 'excel': './results/processed_Walker2d_static.xlsx'},
    {'env': 'Walker-v4', 'type': 'dynamic', 'excel': './results/processed_Walker2d_dynamic.xlsx'},
    {'env': 'Humanoid-v4', 'type': 'basic', 'excel': './results/processed_Humanoid_basic.xlsx'},
    {'env': 'Humanoid-v4', 'type': 'static', 'excel': './results/processed_Humanoid_static.xlsx'},
    {'env': 'Humanoid-v4', 'type': 'dynamic', 'excel': './results/processed_Humanoid_dynamic.xlsx'},
]

def safe_calculate_metrics(avg_df):
    if avg_df is None:
        return [[None, None, None, None]] * 4, [[None, None, None, None]] * 4
    return get_best_models_by_normalized(avg_df)

def get_best_models_by_normalized(avg_df):
    checkpoint_str = 'Checkpoint'
    result = avg_df.groupby(['Model', checkpoint_str]).agg({'Steps': 'sum', 'Reward Divided by Steps': 'sum'}).reset_index()
    result['Distance'] = result['Steps'] * result['Reward Divided by Steps']
    
    return (
        [last_model_speed(result, checkpoint_str), best_step_speed(result), best_speed_speed(result), best_distance_speed(result)],
        [last_model_distance(result, checkpoint_str), best_step_distance(result), best_speed_distance(result), best_distance_distance(result)]
    )

def last_model_speed(result, checkpoint_str):
    filtered_df = result[result[checkpoint_str] == result[checkpoint_str].max()].groupby('Model')
    return calculate_stats(filtered_df, 'Reward Divided by Steps')

def last_model_distance(result, checkpoint_str):
    filtered_df = result[result[checkpoint_str] == result[checkpoint_str].max()].groupby('Model')
    return calculate_stats(filtered_df, 'Distance')

def best_step_speed(result):
    return calculate_stats_by_group(result, 'Steps', 'Reward Divided by Steps')

def best_step_distance(result):
    return calculate_stats_by_group(result, 'Steps', 'Distance')

def best_speed_speed(result):
    return calculate_stats_by_group(result, 'Reward Divided by Steps', 'Reward Divided by Steps')

def best_speed_distance(result):
    return calculate_stats_by_group(result, 'Reward Divided by Steps', 'Distance')

def best_distance_speed(result):
    return calculate_stats_by_group(result, 'Distance', 'Reward Divided by Steps')

def best_distance_distance(result):
    return calculate_stats_by_group(result, 'Distance', 'Distance')

def calculate_stats(filtered_df, column):
    try:
        return [
            filtered_df[column].mean().mean(),
            filtered_df[column].median().median(),
            filtered_df[column].min().std()
        ]
    except:
        return [None, None, None]

def calculate_stats_by_group(result, group_col, target_col):
    try:
        max_val_indices = result.groupby('Model')[group_col].idxmax()
        max_val_result = result.loc[max_val_indices]
        max_target_values = result.loc[max_val_result.index, target_col]
        return [max_target_values.mean(), max_target_values.median(), max_target_values.std()]
    except:
        return [None, None, None]

def generate_summary_excel(all_speed, all_distance):
    print(all_speed)
    data = [
        [None, None, None, "Last Model", None, None, "Best Step", None, None, "Best Speed", None, None, "Best Distance", None, None],
        [None, None, None, "avg", "med", "dev", "avg", "med", "dev", "avg", "med", "dev", "avg", "med", "dev"],
        ["Speed", "Hopper", "basic", *flatten(all_speed[0])],
        [None, None, "static", *flatten(all_speed[1])],
        [None, None, "dynamic", *flatten(all_speed[2])],
        [None, "Walker2D", "basic", *flatten(all_speed[3])],
        [None, None, "static", *flatten(all_speed[4])],
        [None, None, "dynamic", *flatten(all_speed[5])],
        [None, "Humanoid", "basic", *flatten(all_speed[6])],
        [None, None, "static", *flatten(all_speed[7])],
        [None, None, "dynamic", *flatten(all_speed[8])],
        ["Distance", "Hopper", "basic", *flatten(all_distance[0])],
        [None, None, "static", *flatten(all_distance[1])],
        [None, None, "dynamic", *flatten(all_distance[2])],
        [None, "Walker2d", "basic", *flatten(all_distance[3])],
        [None, None, "static", *flatten(all_distance[4])],
        [None, None, "dynamic", *flatten(all_distance[5])],
        [None, "Humanoid", "basic", *flatten(all_distance[6])],
        [None, None, "static", *flatten(all_distance[7])],
        [None, None, "dynamic", *flatten(all_distance[8])],
    ]
    df = pd.DataFrame(data)
    with pd.ExcelWriter("trpo_summary.xlsx", engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, header=False, sheet_name='Sheet1')

def flatten(data):
    return [item for sublist in data for item in (sublist if sublist is not None else [None, None, None])]

if __name__ == "__main__":
    all_speed = []
    all_distance = []

    for datum in data:
        if os.path.exists(datum['excel']):
            avg_df = pd.read_excel(datum['excel'], sheet_name='Averages')
        else:
            avg_df = None
        speeds, distances = safe_calculate_metrics(avg_df)
        all_speed.append(speeds)
        all_distance.append(distances)

    generate_summary_excel(all_speed, all_distance)
