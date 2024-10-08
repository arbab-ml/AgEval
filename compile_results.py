import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import pickle

def calculate_f1(df, shots, method):
    true_labels = df['1'].fillna('Unknown')
    pred_labels = df[f'{method} # of Shots {shots}'].fillna('NA_placeholder')
    return f1_score(true_labels, pred_labels, average='weighted') * 100

def calculate_avg_same_category(df, shots, method):
    return df[f'{method} Same Category Count {shots}'].mean()

def process_model_csvs(results_folder):
    results = {shots: [] for shots in [0, 1, 2, 4, 8]}
    
    for model in os.listdir(results_folder):
        model_path = os.path.join(results_folder, model)
        if os.path.isdir(model_path):
            for encoder in os.listdir(model_path):
                encoder_path = os.path.join(model_path, encoder)
                if os.path.isdir(encoder_path):
                    for file_name in os.listdir(encoder_path):
                        if file_name.endswith('.csv'):
                            dataset_name = os.path.splitext(file_name)[0]
                            file_path = os.path.join(encoder_path, file_name)
                            df = pd.read_csv(file_path)
                            
                            for shots in [0, 1, 2, 4, 8]:
                                for method in ['Embedding', 'Random']:
                                    f1 = calculate_f1(df, shots, method)
                                    avg_same_category = calculate_avg_same_category(df, shots, method)
                                    
                                    results[shots].append({
                                        'Model': model,
                                        'Dataset': dataset_name,
                                        'Method': method,
                                        'Encoder': encoder,
                                        'F1': f1,
                                        'Avg_Same_Category': avg_same_category
                                    })

    return results

# Main execution
if __name__ == "__main__":
    results_folder = 'results'
    analysis_folder = 'results_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    results_dict = process_model_csvs(results_folder)
    
    # Convert results to DataFrames
    result_table_dict = {}
    for shots, data in results_dict.items():
        result_df = pd.DataFrame(data)
        result_df = result_df.pivot_table(values=['F1', 'Avg_Same_Category'], index=['Model', 'Dataset'], columns=['Method', 'Encoder'])
          # Round to two decimal places
        
        # Calculate average of all datasets
        avg_row = result_df.mean()
        avg_df = pd.DataFrame(avg_row).T
        avg_df.index = pd.MultiIndex.from_tuples([('GPT-4o', 'Average')], names=['Model', 'Dataset'])
        result_df = pd.concat([result_df, avg_df])
        result_df = result_df.round(2)
        result_table_dict[shots] = result_df

    # Save the result_table_dict as a pickle file
    with open(os.path.join(analysis_folder, 'result_table_dict.pkl'), 'wb') as f:
        pickle.dump(result_table_dict, f)

    print(f"Results saved in '{analysis_folder}/result_table_dict.pkl'")

    # # Print DataFrames for each shot
    # for shots, df in result_table_dict.items():
    #     print(f"\nResults for {shots} shots:")
    #     print(df)

    # Save result_table_dict for Avg_Same_Category and F1 as text files
    for metric in ['Avg_Same_Category', 'F1']:
        with open(os.path.join('writing/66f622264889b97fcfa0bc72/', f'{metric.lower()}.txt'), 'w') as f:
            for shots, df in result_table_dict.items():
                f.write(f"Results for {shots} shots:\n")
                f.write(df[metric].to_string())
                f.write("\n\n")
