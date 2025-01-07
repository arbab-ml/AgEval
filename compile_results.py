import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import pickle

TAXONOMIC_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def calculate_f1(df, shots, method, level):
    # Extract true labels from hierarchy dictionary for the given level
    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
    pred_labels = df[f'{method} {level} {shots}'].fillna('NA_placeholder')
    return f1_score(true_labels, pred_labels, average='weighted') * 100

def calculate_avg_same_category(df, shots, method, level):
    # Get true label for each image at the given level
    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
    
    # Get example categories for each image
    example_categories = df[f'{method} Example Categories {level} {shots}'].apply(lambda x: eval(x) if isinstance(x, str) and x.strip('[]') else [])
    
    # Count how many examples match the true label
    matches = []
    for true_label, example_list in zip(true_labels, example_categories):
        match_count = sum(1 for ex in example_list if ex == true_label)
        matches.append(match_count)
    
    return np.mean(matches)

def process_model_csvs(results_folder):
    results = {shots: [] for shots in [1]}  # Currently only processing 1-shot
    
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
                            
                            try:
                                df = pd.read_csv(file_path)
                                
                                for shots in [1]:  # Currently only processing 1-shot
                                    for method in ['Embedding', 'Random']:
                                        for level in TAXONOMIC_LEVELS:
                                            try:
                                                f1 = calculate_f1(df, shots, method, level)
                                                avg_same_category = calculate_avg_same_category(df, shots, method, level)
                                                
                                                results[shots].append({
                                                    'Model': model,
                                                    'Dataset': dataset_name,
                                                    'Method': method,
                                                    'Encoder': encoder,
                                                    'Level': level,
                                                    'F1': f1,
                                                    'Avg_Same_Category': avg_same_category
                                                })
                                            except Exception as e:
                                                print(f"Error processing {method} for {dataset_name} at {level}: {str(e)}")
                            except Exception as e:
                                print(f"Error reading file {file_path}: {str(e)}")

    return results

# Main execution
if __name__ == "__main__":
    results_folder = 'results-hierarchical'
    analysis_folder = 'results-hierarchical-analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    results_dict = process_model_csvs(results_folder)
    
    # Convert results to DataFrames
    result_table_dict = {}
    for shots, data in results_dict.items():
        result_df = pd.DataFrame(data)
        if not result_df.empty:
            # Create separate pivot tables for each taxonomic level
            level_tables = {}
            for level in TAXONOMIC_LEVELS:
                level_df = result_df[result_df['Level'] == level].copy()
                if not level_df.empty:
                    level_pivot = level_df.pivot_table(
                        values=['F1', 'Avg_Same_Category'],
                        index=['Model', 'Dataset'],
                        columns=['Method', 'Encoder']
                    )
                    
                    # Calculate average of all datasets
                    avg_row = level_pivot.mean()
                    avg_df = pd.DataFrame(avg_row).T
                    avg_df.index = pd.MultiIndex.from_tuples([('Average', 'All Datasets')], names=['Model', 'Dataset'])
                    level_pivot = pd.concat([level_pivot, avg_df])
                    level_pivot = level_pivot.round(2)
                    level_tables[level] = level_pivot
            
            result_table_dict[shots] = level_tables

    # Save the result_table_dict as a pickle file
    with open(os.path.join(analysis_folder, 'hierarchical_result_table_dict.pkl'), 'wb') as f:
        pickle.dump(result_table_dict, f)

    print(f"Results saved in '{analysis_folder}/hierarchical_result_table_dict.pkl'")

    # Print DataFrames for each shot and level
    for shots, level_tables in result_table_dict.items():
        print(f"\nResults for {shots} shots:")
        for level, df in level_tables.items():
            print(f"\n{level.capitalize()} Level:")
            print(df)

    # Save results as text files, one for each metric and level
    for metric in ['F1', 'Avg_Same_Category']:
        with open(os.path.join(analysis_folder, f'hierarchical_{metric.lower()}_results.txt'), 'w') as f:
            for shots, level_tables in result_table_dict.items():
                f.write(f"Results for {shots} shots:\n")
                for level, df in level_tables.items():
                    f.write(f"\n{level.capitalize()} Level:\n")
                    f.write(df[metric].to_string())
                    f.write("\n\n")
