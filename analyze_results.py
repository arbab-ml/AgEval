import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the style for the plots
plt.style.use('ggplot')

# Function to calculate accuracy
def calc_accuracy(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()

# Function to process a single CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic information about the dataset
    print(f"\nProcessing: {os.path.basename(file_path)}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize a list to store results
    results = []

    # Calculate accuracy for all shot counts
    shot_counts = [0, 1, 2, 4, 8]
    for shots in shot_counts:
        for method in ['Embedding', 'Random']:
            column_name = f"{method} # of Shots {shots}"
            
            # Check for any mismatched predictions
            mismatched_count = (df['1'] != df[column_name]).sum()
            
            # Calculate accuracy
            accuracy = calc_accuracy(df['1'], df[column_name])
            
            # Calculate average same category count
            same_category_count = df[f"{method} Same Category Count {shots}"].mean()
            
            # Store results
            results.append({
                'Shots': shots,
                'Method': method,
                'Mismatched_Count': mismatched_count,
                'Accuracy': accuracy,
                'Avg_Same_Category_Count': same_category_count
            })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Generate plots
    generate_plots(results_df, os.path.basename(file_path))

    # Additional analysis
    print("\nDistribution of true labels:")
    print(df['1'].value_counts(normalize=True))

    # Check for any missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    return results_df

# Function to generate plots
def generate_plots(results_df, file_name):
    plots_folder = os.path.join('results_analysis', 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Plot 1: Accuracy vs Number of Shots
    plt.figure(figsize=(10, 6))
    for method in ['Embedding', 'Random']:
        method_data = results_df[results_df['Method'] == method]
        plt.plot(method_data['Shots'], method_data['Accuracy'], marker='o', label=method)

    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Number of Shots - {file_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'accuracy_vs_shots_{file_name}.png'))
    plt.close()

    # Plot 2: Average Same Category Count vs Number of Shots
    plt.figure(figsize=(10, 6))
    for method in ['Embedding', 'Random']:
        method_data = results_df[results_df['Method'] == method]
        plt.plot(method_data['Shots'], method_data['Avg_Same_Category_Count'], marker='o', label=method)

    plt.xlabel('Number of Shots')
    plt.ylabel('Average Same Category Count')
    plt.title(f'Average Same Category Count vs Number of Shots - {file_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'avg_same_category_count_vs_shots_{file_name}.png'))
    plt.close()

    print(f"Plots saved for {file_name} in the '{plots_folder}' folder.")

# Main execution
if __name__ == "__main__":
    input_folder = 'results/GPT-4o'
    results_folder = 'results_analysis'
    os.makedirs(results_folder, exist_ok=True)

    # Process each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            results_df = process_csv(file_path)
            
            # Save individual results CSV
            dataset_name = os.path.splitext(file_name)[0]
            individual_results_path = os.path.join(results_folder, f'{dataset_name}_accuracy_results.csv')
            results_df.to_csv(individual_results_path, index=False)
            print(f"Results for {dataset_name} saved in '{individual_results_path}'")

    print(f"\nAll individual results saved in '{results_folder}' folder.")