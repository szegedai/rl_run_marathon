import pandas as pd
import os

data = [
         {  'env': 'Hopper-v4',
            'type': 'basic',
            'excel': './evaluate/results/baseline_hopper.xlsx',
        },
         {
            'env': 'Hopper-v4',
            'type': 'static',
            'excel': './evaluate/results/static_hopper.xlsx',
         },
        {  'env': 'Hopper-v4',
            'type': 'dynamic',
            'max_ep': "25200",
            'excel': './evaluate/results/dynamic_hopper.xlsx',
        },
         ]


def avg(input_excel_path, type):
    df = pd.read_excel(input_excel_path)
    grouped = df.groupby('Model').mean().reset_index()

    result_df = pd.DataFrame(grouped)
    result_df = result_df.drop(columns=['Unnamed: 0', 'Seed'], errors='ignore')
    result_df = separate_checkpoint(result_df)

    if (type == 'dynamic'):
        result_df = result_df.groupby('Model', group_keys=False).apply(map_ids)

    return result_df


def map_ids(group):
    unique_ids = sorted(group['Checkpoint'].astype(int).unique())
    id_map = {val: idx for idx, val in enumerate(unique_ids)}
    group['Checkpoint'] = group['Checkpoint'].astype(int).map(id_map)
    return group


def separate_checkpoint(df):
    if not {'Checkpoint', 'Model'}.issubset(df.columns):
        model_index = df.columns.get_loc('Model')
        df.insert(model_index, 'Checkpoint', df['Model'].str.split('/').str[1])
        df['Model'] = df['Model'].str.split('/').str[0]
    return df


def save_averages(excel_name, result_df, df):
    output_excel_path = excel_name
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        result_df.to_excel(writer, sheet_name='Averages', index=False)


if __name__ == "__main__":
    if not os.path.exists('averages'):
        os.makedirs('averages')

    for datum in data:
        input_excel_path = datum['excel']

        df = pd.read_excel(input_excel_path)
        df = separate_checkpoint(df)
        print(datum['env'])
        print(datum['type'])
        result_df = avg(input_excel_path, datum['type'])
        result_df['Checkpoint'] = pd.to_numeric(result_df['Checkpoint'], errors='coerce')

        averages_excel_name = 'averages/processed_'+datum['env']+'_'+datum['type']+'.xlsx'
        save_averages(averages_excel_name, result_df, df)
