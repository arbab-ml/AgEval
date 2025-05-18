import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import pickle

TAXONOMIC_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
SHOTS_TO_PROCESS = [ 1, 8]  # Process both 1-shot and 8-shot results

def calculate_f1(df, shots, method, level):
    """
    Calculate F1 score for a specific taxonomic level and count different types of errors.
    
    This function identifies cascade errors during analysis:
    - If any higher taxonomic level has an error, it's considered a cascade error
    - This approach allows inference.py to be simpler, with no explicit cascade error marking
    - For embedding mode, inference.py skips lower levels after errors
    - For random mode, inference.py continues to predict lower levels
    
    Args:
        df: DataFrame containing predictions and true labels
        shots: Number of shots (examples) used
        method: 'Embedding' or 'Random'
        level: Taxonomic level to evaluate
        
    Returns:
        F1 score and counts of different error types
    """
    # Filter for evaluated rows
    if 'evaluated' in df.columns:
        df = df[df['evaluated']]
    
    # Extract labels and predictions
    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
    pred_column = f'{method} {level} {shots}'
    predictions = df[pred_column]
    
    # Define error codes and count them
    error_types = {
        'NA_JSON_PARSE': 'JSON Parsing Errors',
        'NA_INVALID_PRED': 'Invalid Predictions',
        'NA_API_ERROR': 'API Errors',
        'NA_GENERAL': 'General Errors',
        'NA_CASCADE': 'Cascading Errors from Higher Levels'
    }
    
    # Count direct errors at this level
    error_counts = {error_code: 0 for error_code in error_types.keys()}
    
    # Check for cascading errors from higher levels
    taxonomic_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    current_level_idx = taxonomic_order.index(level)
    
    for idx in df.index:
        # Check higher levels for errors that would cascade
        has_cascade_error = False
        for higher_level in taxonomic_order[:current_level_idx]:
            higher_pred = df.at[idx, f'{method} {higher_level} {shots}']
            if str(higher_pred).startswith('NA_'):
                has_cascade_error = True
                break
        
        if has_cascade_error:
            # Count as cascade error, even if there's a direct prediction or error
            error_counts['NA_CASCADE'] += 1
        else:
            # Only count direct errors if there's no cascade
            pred = predictions[idx]
            if str(pred).startswith('NA_'):
                error_type = str(pred)
                if error_type in error_counts:
                    error_counts[error_type] += 1
    
    # Filter out error codes for F1 calculation
    valid_mask = ~predictions.isin(error_types.keys())
    valid_predictions = predictions[valid_mask]
    valid_true_labels = true_labels[valid_mask]
    
    # Calculate F1 score
    try:
        if len(valid_predictions) > 0:
            valid_true_labels = valid_true_labels.astype(str)
            valid_predictions = valid_predictions.astype(str)
            f1 = f1_score(valid_true_labels, valid_predictions, average='weighted') * 100
        else:
            f1 = 0.0
    except Exception as e:
        f1 = 0.0
    
    return f1, error_counts

def calculate_avg_same_category(df, shots, method, level):
    # Filter for evaluated rows if the column exists
    if 'evaluated' in df.columns:
        df = df[df['evaluated']]
    
    # Get true label for each image at the given level
    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
    
    # Get example categories for each image
    example_categories = df[f'{method} Example Categories {level} {shots}'].apply(lambda x: eval(x) if isinstance(x, str) and x.strip('[]') else [])
    
    # Count how many examples match the true label
    matches = []
    for true_label, example_list in zip(true_labels, example_categories):
        match_count = sum(1 for ex in example_list if ex == true_label)
        matches.append(match_count / len(example_list) if example_list else 0)
    
    return np.mean(matches) * 100

def process_model_csvs(results_folder):
    results = {shots: [] for shots in SHOTS_TO_PROCESS}
    
    target_file = os.path.join(results_folder, "GPT-4o-mini", "vit", "BioTrove-Balanced_219species_10samples_eval10pct.csv")
    
    if os.path.exists(target_file):
        try:
            df = pd.read_csv(target_file)
            dataset_name = os.path.splitext(os.path.basename(target_file))[0]
            
            for shots in SHOTS_TO_PROCESS:
                for method in ['Embedding', 'Random']:
                    for level in TAXONOMIC_LEVELS:
                        try:
                            f1, error_counts = calculate_f1(df, shots, method, level)
                            avg_same_category = calculate_avg_same_category(df, shots, method, level)
                            
                            results[shots].append({
                                'Model': 'GPT-4o-mini',
                                'Dataset': dataset_name,
                                'Method': method,
                                'Encoder': 'vit',
                                'Level': level,
                                'F1': f1,
                                'Avg_Same_Category': avg_same_category,
                                'NA_JSON_PARSE': error_counts.get('NA_JSON_PARSE', 0),
                                'NA_INVALID_PRED': error_counts.get('NA_INVALID_PRED', 0),
                                'NA_API_ERROR': error_counts.get('NA_API_ERROR', 0),
                                'NA_GENERAL': error_counts.get('NA_GENERAL', 0),
                                'NA_CASCADE': error_counts.get('NA_CASCADE', 0)
                            })
                        except Exception as e:
                            print(f"Error processing {method} for {dataset_name} at {level} with {shots} shots: {str(e)}")
        except Exception as e:
            print(f"Error reading file {target_file}: {str(e)}")
    else:
        print(f"Target file not found: {target_file}")

    return results

def print_results_table(result_table_dict):
    print("\nResults Summary")
    print("=" * 120)
    
    for shots, level_tables in result_table_dict.items():
        print(f"\n{shots}-Shot Results")
        print("-" * 120)
        print(f"{'Level':<10} | {'STAGE F1':>12} | {'Random F1':>10} | {'STAGE Examples':>16} | {'Random Examples':>13} | {'STAGE Errors':>25} | {'Random Errors':>25}")
        print("-" * 120)
        
        for level in TAXONOMIC_LEVELS:
            try:
                if level in level_tables:
                    df = level_tables[level]
                    avg_row = df.iloc[-1]
                    
                    stage_f1 = avg_row.get(('F1', 'Embedding', 'vit'), 0.0)
                    rand_f1 = avg_row.get(('F1', 'Random', 'vit'), 0.0)
                    stage_matches = avg_row.get(('Avg_Same_Category', 'Embedding', 'vit'), 0.0)
                    rand_matches = avg_row.get(('Avg_Same_Category', 'Random', 'vit'), 0.0)
                    
                    # Get error counts for each type
                    error_types = ['NA_JSON_PARSE', 'NA_INVALID_PRED', 'NA_API_ERROR', 'NA_GENERAL', 'NA_CASCADE']
                    stage_errors = []
                    random_errors = []
                    for err_type in error_types:
                        stage_count = int(avg_row.get((err_type, 'Embedding', 'vit'), 0))
                        rand_count = int(avg_row.get((err_type, 'Random', 'vit'), 0))
                        if stage_count > 0:
                            stage_errors.append(f"{err_type[3:]}({stage_count})")
                        if rand_count > 0:
                            random_errors.append(f"{err_type[3:]}({rand_count})")
                    
                    stage_error_str = ", ".join(stage_errors) if stage_errors else "None"
                    random_error_str = ", ".join(random_errors) if random_errors else "None"
                else:
                    stage_f1 = rand_f1 = stage_matches = rand_matches = 0.0
                    stage_error_str = random_error_str = "N/A"
                
                print(f"{level.capitalize():<10} | {stage_f1:>12.2f} | {rand_f1:>10.2f} | {stage_matches:>16.2f} | {rand_matches:>13.2f} | {stage_error_str:>25} | {random_error_str:>25}")
            except Exception as e:
                print(f"{level.capitalize():<10} | {0:>12.2f} | {0:>10.2f} | {0:>16.2f} | {0:>13.2f} | {'Error':>25} | {'Error':>25}")
        print("-" * 120)

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
                        values=['F1', 'Avg_Same_Category', 'NA_JSON_PARSE', 'NA_INVALID_PRED', 
                               'NA_API_ERROR', 'NA_GENERAL', 'NA_CASCADE'],
                        index=['Model', 'Dataset'],
                        columns=['Method', 'Encoder']
                    )
                    level_pivot = level_pivot.round(2)
                    level_tables[level] = level_pivot
            
            result_table_dict[shots] = level_tables

    # Save the result_table_dict as a pickle file
    with open(os.path.join(analysis_folder, 'hierarchical_result_table_dict.pkl'), 'wb') as f:
        pickle.dump(result_table_dict, f)

    # print(f"\nResults for BioTrove-Balanced_219species_10samples_eval2pct.csv")
    print("=" * 100)
    
    # Print formatted results
    print_results_table(result_table_dict)

    # Save results as text files
    for metric in ['F1', 'Avg_Same_Category']:
        output_file = os.path.join(analysis_folder, f'hierarchical_{metric.lower()}_results.txt')
        with open(output_file, 'w') as f:
            # f.write(f"Results for BioTrove-Balanced_219species_10samples_eval2pct.csv\n")
            f.write("=" * 100 + "\n")
            
            for shots, level_tables in result_table_dict.items():
                f.write(f"\n{shots}-Shot Results\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Level':<10} | {'STAGE':>12} | {'Random':>10}\n")
                f.write("-" * 100 + "\n")
                
                for level, df in level_tables.items():
                    try:
                        stage_val = df.iloc[0].get((metric, 'Embedding', 'vit'), 0.0)
                        rand_val = df.iloc[0].get((metric, 'Random', 'vit'), 0.0)
                        f.write(f"{level.capitalize():<10} | {stage_val:>12.2f} | {rand_val:>10.2f}\n")
                    except Exception as e:
                        f.write(f"{level.capitalize():<10} | {'N/A':>12} | {'N/A':>10}\n")
                f.write("-" * 100 + "\n\n")
