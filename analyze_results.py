import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the style for the plots
plt.style.use('ggplot')  # Using 'ggplot' style instead of 'seaborn'

# Create a results folder if it doesn't exist
results_folder = 'results_analysis'
plots_folder = os.path.join(results_folder, 'plots')
os.makedirs(plots_folder, exist_ok=True)

# Read the results
results_df = pd.read_csv(os.path.join(results_folder, 'accuracy_results.csv'))
true_label_distribution = pd.read_csv(os.path.join(results_folder, 'true_label_distribution.csv'), index_col=0).squeeze()
missing_values = pd.read_csv(os.path.join(results_folder, 'missing_values.csv'), index_col=0).squeeze()

# Plot 1: Accuracy vs Number of Shots
plt.figure(figsize=(10, 6))
for method in ['Embedding', 'Random']:
    method_data = results_df[results_df['Method'] == method]
    plt.plot(method_data['Shots'], method_data['Accuracy'], marker='o', label=method)

plt.xlabel('Number of Shots')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Shots')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'accuracy_vs_shots.png'))
plt.close()

# Plot 2: Average Same Category Count vs Number of Shots
plt.figure(figsize=(10, 6))
for method in ['Embedding', 'Random']:
    method_data = results_df[results_df['Method'] == method]
    plt.plot(method_data['Shots'], method_data['Avg_Same_Category_Count'], marker='o', label=method)

plt.xlabel('Number of Shots')
plt.ylabel('Average Same Category Count')
plt.title('Average Same Category Count vs Number of Shots')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'avg_same_category_count_vs_shots.png'))
plt.close()

# Plot 3: True Label Distribution
plt.figure(figsize=(10, 6))
plt.bar(true_label_distribution.index, true_label_distribution.values)
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.title('Distribution of True Labels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'true_label_distribution.png'))
plt.close()

# Plot 4: Mismatched Count vs Number of Shots
plt.figure(figsize=(10, 6))
for method in ['Embedding', 'Random']:
    method_data = results_df[results_df['Method'] == method]
    plt.plot(method_data['Shots'], method_data['Mismatched_Count'], marker='o', label=method)

plt.xlabel('Number of Shots')
plt.ylabel('Mismatched Count')
plt.title('Mismatched Count vs Number of Shots')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'mismatched_count_vs_shots.png'))
plt.close()

print(f"Plots saved in the '{plots_folder}' folder.")