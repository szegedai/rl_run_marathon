import pandas as pd
import os

data = [
        {
            'env': 'Hopper',
            'type': 'baseline_sgld',
            'excel': 'default_evaluate_folder/sgld_baseline.xlsx',
        },
        {
            'env': 'Hopper',
            'type': 'dynamic_sgld',
            'excel': 'default_evaluate_folder/sgld_dynamic.xlsx',
        },
        {
            'env': 'Hopper',
            'type': 'baseline_sppo',
            'excel': 'default_evaluate_folder/vanilla_baseline.xlsx',
        },
        {
            'env': 'Hopper',
            'type': 'dynamic_sppo',
            'excel': 'default_evaluate_folder/vanilla_dynamic.xlsx',
        }
    ]


def avg(input_excel_path):
    df = pd.read_excel(input_excel_path)
    grouped = df.groupby(['Model', 'RowId']).mean().reset_index()

    result_df = pd.DataFrame(grouped)
    result_df = result_df.drop(columns=['Unnamed: 0', 'Episode'], errors='ignore')

    return result_df


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
        print(datum['env'])
        print(datum['type'])
        result_df = avg(input_excel_path)
        result_df['RowId'] = pd.to_numeric(result_df['RowId'], errors='coerce')

        averages_excel_name = 'averages/processed_'+datum['env']+'_'+datum['type']+'.xlsx'
        save_averages(averages_excel_name, result_df, df)
