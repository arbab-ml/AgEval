import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import pickle

TAXONOMIC_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
SHOTS_TO_PROCESS = [1, 8]  # Process both 1-shot and 8-shot results

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
    results = {shots: [] for shots in SHOTS_TO_PROCESS}
    
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
                                
                                for shots in SHOTS_TO_PROCESS:
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
                                                print(f"Error processing {method} for {dataset_name} at {level} with {shots} shots: {str(e)}")
                            except Exception as e:
                                print(f"Error reading file {file_path}: {str(e)}")

    return results

def print_results_table(result_table_dict):
    print("\nResults Summary")
    print("=" * 100)
    
    for shots, level_tables in result_table_dict.items():
        print(f"\n{shots}-Shot Results")
        print("-" * 100)
        print(f"{'Level':<10} | {'Embedding F1':>12} | {'Random F1':>10} | {'Embedding Matches':>16} | {'Random Matches':>13}")
        print("-" * 100)
        
        for level, df in level_tables.items():
            # Get average values (last row of each dataframe)
            avg_row = df.iloc[-1]
            emb_f1 = avg_row[('F1', 'Embedding', 'vit')]
            rand_f1 = avg_row[('F1', 'Random', 'vit')]
            emb_matches = avg_row[('Avg_Same_Category', 'Embedding', 'vit')]
            rand_matches = avg_row[('Avg_Same_Category', 'Random', 'vit')]
            
            print(f"{level.capitalize():<10} | {emb_f1:>12.2f} | {rand_f1:>10.2f} | {emb_matches:>16.2f} | {rand_matches:>13.2f}")
        print("-" * 100)

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
    
    # Print formatted results
    print_results_table(result_table_dict)

    # Save results as text files, one for each metric and level
    for metric in ['F1', 'Avg_Same_Category']:
        with open(os.path.join(analysis_folder, f'hierarchical_{metric.lower()}_results.txt'), 'w') as f:
            for shots, level_tables in result_table_dict.items():
                f.write(f"\n{shots}-Shot Results\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Level':<10} | {'Embedding':>12} | {'Random':>10}\n")
                f.write("-" * 100 + "\n")
                
                for level, df in level_tables.items():
                    avg_row = df.iloc[-1]
                    emb_val = avg_row[(metric, 'Embedding', 'vit')]
                    rand_val = avg_row[(metric, 'Random', 'vit')]
                    f.write(f"{level.capitalize():<10} | {emb_val:>12.2f} | {rand_val:>10.2f}\n")
                f.write("-" * 100 + "\n\n")
