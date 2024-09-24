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
def generate_plots(results_df, model_name, dataset_name):
    plots_folder = os.path.join('results_analysis', 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Define a color cycle
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(results_df['Encoder'].unique())))

    # Plot 1: Accuracy vs Number of Shots
    plt.figure(figsize=(12, 8))
    for i, encoder in enumerate(results_df['Encoder'].unique()):
        for method in ['Embedding', 'Random']:
            data = results_df[(results_df['Encoder'] == encoder) & (results_df['Method'] == method)]
            linestyle = '-' if method == 'Embedding' else ':'
            plt.plot(data['Shots'], data['Accuracy'], marker='o', linestyle=linestyle, color=colors[i], label=f'{encoder} - {method}')

    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Number of Shots - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'accuracy_vs_shots_{model_name}_{dataset_name}.png'))
    plt.close()

    # Plot 2: Average Same Category Count vs Number of Shots
    plt.figure(figsize=(12, 8))
    for i, encoder in enumerate(results_df['Encoder'].unique()):
        for method in ['Embedding', 'Random']:
            data = results_df[(results_df['Encoder'] == encoder) & (results_df['Method'] == method)]
            linestyle = '-' if method == 'Embedding' else ':'
            plt.plot(data['Shots'], data['Avg_Same_Category_Count'], marker='o', linestyle=linestyle, color=colors[i], label=f'{encoder} - {method}')

    plt.xlabel('Number of Shots')
    plt.ylabel('Average Same Category Count')
    plt.title(f'Average Same Category Count vs Number of Shots - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'avg_same_category_count_vs_shots_{model_name}_{dataset_name}.png'))
    plt.close()

    print(f"Plots saved for {model_name} in the '{plots_folder}' folder.")

# Main execution
if __name__ == "__main__":
    results_folder = 'results'
    analysis_folder = 'results_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    for model_name in os.listdir(results_folder):
        model_folder = os.path.join(results_folder, model_name)
        if os.path.isdir(model_folder):
            print(f"\nProcessing model: {model_name}")
            results_df = process_model_csvs(model_folder)
            
            # Save combined results CSV for the model
            combined_results_path = os.path.join(analysis_folder, f'{model_name}_combined_results.csv')
            results_df.to_csv(combined_results_path, index=False)
            print(f"Combined results for {model_name} saved in '{combined_results_path}'")
            
            # Generate plots for the model
            dataset_name = os.path.splitext(os.listdir(model_folder)[0])[0]  # Infer dataset name from the first CSV file
            generate_plots(results_df, model_name, dataset_name)

    print(f"\nAll results and plots saved in '{analysis_folder}' folder.")