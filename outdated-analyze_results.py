import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the style for the plots
plt.style.use('ggplot')

# Function to calculate accuracy
def calc_accuracy(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()

# Function to process CSV files for a specific model
def process_model_csvs(model_folder):
    results = []
    
    for encoder_folder in os.listdir(model_folder):
        encoder_path = os.path.join(model_folder, encoder_folder)
        if os.path.isdir(encoder_path):
            for file_name in os.listdir(encoder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(encoder_path, file_name)
                    df = pd.read_csv(file_path)
                    
                    print(f"\nProcessing: {encoder_folder}/{file_name}")
                    print(f"Total samples: {len(df)}")
                    print(f"Columns: {df.columns.tolist()}")

                    shot_counts = [0, 1, 2, 4, 8]
                    for shots in shot_counts:
                        for method in ['Embedding', 'Random']:
                            column_name = f"{method} # of Shots {shots}"
                            
                            mismatched_count = (df['1'] != df[column_name]).sum()
                            accuracy = calc_accuracy(df['1'], df[column_name])
                            same_category_count = df[f"{method} Same Category Count {shots}"].mean()
                            
                            results.append({
                                'Encoder': encoder_folder,
                                'Dataset': os.path.splitext(file_name)[0],
                                'Shots': shots,
                                'Method': method,
                                'Mismatched_Count': mismatched_count,
                                'Accuracy': accuracy,
                                'Avg_Same_Category_Count': same_category_count
                            })

    return pd.DataFrame(results)

# Function to generate plots
def generate_plots(results_df, model_name):
    plots_folder = os.path.join('results_analysis', 'plots', model_name)
    os.makedirs(plots_folder, exist_ok=True)

    # Define a color cycle
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(results_df['Encoder'].unique())))

    for dataset_name in results_df['Dataset'].unique():
        dataset_results = results_df[results_df['Dataset'] == dataset_name]

        # Plot 1: Accuracy vs Number of Shots
        plt.figur