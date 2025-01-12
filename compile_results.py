import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import pickle

TAXONOMIC_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
SHOTS_TO_PROCESS = [1, 8]  # Process both 1-shot and 8-shot results

def calculate_f1(df, shots, method, level):
    # Filter for evaluated rows if the column exists
    if 'evaluated' in df.columns:
        df = df[df['evaluated']]
        print(f"\nTotal evaluated rows: {len(df)}")
    
    # Extract true labels from hierarchy dictionary for the given level
    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
    pred_labels = df[f'{method} {level} {shots}'].fillna('NA_placeholder')
    
    # Print detailed analysis for kingdom level
    if level == 'kingdom':
        print(f"\nDetailed Analysis for {method} {level} {shots}-shot:")
        print("True label distribution:")
        print(true_labels.value_counts())
        print("\nPredicted label distribution:")
        print(pred_labels.value_counts())
        
        # Add verification of predictions
        print("\nVerification of predictions:")
        print(f"Number of evaluated rows: {len(df)}")
        print(f"Number of predictions: {len(pred_labels)}")
        print(f"Number of non-NA predictions: {len(pred_labels[pred_labels != 'NA_placeholder'])}")
        
        # Print some example rows
        print("\nExample rows with predictions:")
        sample_df = df.sample(min(5, len(df)))
        for idx, row in sample_df.iterrows():
            hierarchy = eval(row['hierarchy'])
            pred = row[f'{method} {level} {shots}']
            print(f"\nRow {idx}:")
            print(f"True label: {hierarchy[level]}")
            print(f"Predicted: {pred}")
            print(f"Evaluated: {row['evaluated']}")
        
        print("\nConfusion Matrix:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print(cm)
    
    return f1_score(true_labels, pred_labels, average='weighted') * 100

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
    
    # Specific file path
    target_file = os.path.join(results_folder, "GPT-4o-mini", "vit", "BioTrove-Balanced_219species_10samples_eval10pct.csv")
    
    if os.path.exists(target_file):
        try:
            df = pd.read_csv(target_file)
            dataset_name = os.path.splitext(os.path.basename(target_file))[0]
            
            # Print dataset statistics
            print("\nDataset Statistics:")
            print("=" * 50)
            print(f"Total rows: {len(df)}")
            if 'evaluated' in df.columns:
                print(f"Evaluated rows: {df['evaluated'].sum()}")
                print(f"Evaluation percentage: {(df['evaluated'].sum() / len(df)) * 100:.2f}%")
                
                # Print distribution of evaluated rows
                eval_df = df[df['evaluated']]
                print("\nDistribution in evaluated rows:")
                eval_dist = eval_df['hierarchy'].apply(lambda x: eval(x)['kingdom']).value_counts()
                print(eval_dist)
            
            # Print class distribution for each taxonomic level
            for level in TAXONOMIC_LEVELS:
                if 'hierarchy' in df.columns:
                    true_labels = df['hierarchy'].apply(lambda x: eval(x)[level] if isinstance(x, str) else 'Unknown')
                    unique_classes = len(true_labels.unique())
                    print(f"\n{level.capitalize()} level unique classes: {unique_classes}")
                    if level == 'kingdom':
                        print("Kingdom distribution:")
                        print(true_labels.value_counts())
            
            # Print example of predictions for kingdom level
            if 'evaluated' in df.columns:
                eval_df = df[df['evaluated']]
                print("\nSample of Kingdom predictions (first 5 evaluated rows):")
                for idx, row in eval_df.head().iterrows():
                    hierarchy = eval(row['hierarchy'])
                    print(f"\nTrue Kingdom: {hierarchy['kingdom']}")
                    print(f"1-shot Embedding pred: {row['Embedding kingdom 1']}")
                    print(f"8-shot Embedding pred: {row['Embedding kingdom 8']}")
                    print(f"1-shot Random pred: {row['Random kingdom 1']}")
                    print(f"8-shot Random pred: {row['Random kingdom 8']}")
            
            for shots in SHOTS_TO_PROCESS:
                for method in ['Embedding', 'Random']:
                    for level in TAXONOMIC_LEVELS:
                        try:
                            f1 = calculate_f1(df, shots, method, level)
                            avg_same_category = calculate_avg_same_category(df, shots, method, level)
                            
                            results[shots].append({
                                'Model': 'GPT-4o-mini',
                                'Dataset': dataset_name,
                                'Method': method,
                                'Encoder': 'vit',
                                'Level': level,
                                'F1': f1,
                                'Avg_Same_Category': avg_same_category
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
    print("=" * 100)
    
    for shots, level_tables in result_table_dict.items():
        print(f"\n{shots}-Shot Results")
        print("-" * 100)
        print(f"{'Level':<10} | {'STAGE F1':>12} | {'Random F1':>10} | {'STAGE Examples':>16} | {'Random Examples':>13}")
        print("-" * 100)
        
        for level, df in level_tables.items():
            # Get average values (last row of each dataframe)
            avg_row = df.iloc[-1]
            stage_f1 = avg_row[('F1', 'Embedding', 'vit')]  # Use Embedding internally
            rand_f1 = avg_row[('F1', 'Random', 'vit')]
            stage_matches = avg_row[('Avg_Same_Category', 'Embedding', 'vit')]  # Use Embedding internally
            rand_matches = avg_row[('Avg_Same_Category', 'Random', 'vit')]
            
            print(f"{level.capitalize():<10} | {stage_f1:>12.2f} | {rand_f1:>10.2f} | {stage_matches:>16.2f} | {rand_matches:>13.2f}")
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
                    level_pivot = level_pivot.round(2)
                    level_tables[level] = level_pivot
            
            result_table_dict[shots] = level_tables

    # Save the result_table_dict as a pickle file
    with open(os.path.join(analysis_folder, 'hierarchical_result_table_dict.pkl'), 'wb') as f:
        pickle.dump(result_table_dict, f)

    print(f"\nResults for BioTrove-Balanced_219species_10samples_eval10pct.csv")
    print("=" * 100)
    
    # Print formatted results
    print_results_table(result_table_dict)

    # Save results as text files, one for each metric and level
    for metric in ['F1', 'Avg_Same_Category']:
        output_file = os.path.join(analysis_folder, f'hierarchical_{metric.lower()}_results.txt')
        with open(output_file, 'w') as f:
            f.write(f"Results for BioTrove-Balanced_219species_10samples_eval10pct.csv\n")
            f.write("=" * 100 + "\n")
            
            for shots, level_tables in result_table_dict.items():
                f.write(f"\n{shots}-Shot Results\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Level':<10} | {'STAGE':>12} | {'Random':>10}\n")
                f.write("-" * 100 + "\n")
                
                for level, df in level_tables.items():
                    stage_val = df.iloc[0][('F1', 'Embedding', 'vit')]  # Use first row since we only have one dataset
                    rand_val = df.iloc[0][('F1', 'Random', 'vit')]
                    f.write(f"{level.capitalize():<10} | {stage_val:>12.2f} | {rand_val:>10.2f}\n")
                f.write("-" * 100 + "\n\n")
