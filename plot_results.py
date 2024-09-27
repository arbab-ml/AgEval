import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Set up the plot style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
})

def generate_plots(result_table_dict):
    plots_folder = os.path.join('results_analysis', 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Prepare data
    shots = [0, 1, 2, 4, 8]
    datasets = result_table_dict[0].index.get_level_values('Dataset').unique()
    models = result_table_dict[0].index.get_level_values('Model').unique()
    encoders = result_table_dict[0].columns.get_level_values('Encoder').unique()

    # Set up the grid
    n_datasets = len(datasets)
    n_cols = 4  # 4 figures per row
    n_rows = math.ceil(n_datasets / n_cols)

    # Create the main figure
    fig = plt.figure(figsize=(15, 3.2 * n_rows))  # Slightly increased height

    # Use GridSpec for more control over subplot positioning
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4)  # Increased hspace for more space between rows

    # Color palette for encoders
    encoder_colors = {'clip': '#1f77b4', 'resnet': '#ff7f0e', 'vit': '#2ca02c'}

    # Plot each dataset
    for i, dataset in enumerate(datasets):
        row = i // n_cols
        col = i % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        all_values = []
        for model in models:
            for encoder in encoders:
                # Plot Embedding results for all encoders
                embedding_values = [result_table_dict[shot].loc[(model, dataset), ('Embedding', encoder)] for shot in shots]
                all_values.extend(embedding_values)
                ax.plot(shots, embedding_values, marker='o', markersize=3, linewidth=1, 
                        linestyle='-', color=encoder_colors[encoder], 
                        label=f'{model} - {encoder} (Embedding)')
                
                # Plot Random results only for 'clip' encoder
                if encoder == 'clip':
                    random_values = [result_table_dict[shot].loc[(model, dataset), ('Random', encoder)] for shot in shots]
                    all_values.extend(random_values)
                    ax.plot(shots, random_values, marker='o', markersize=3, linewidth=1, 
                            linestyle=':', color=encoder_colors[encoder], 
                            label=f'{model} - {encoder} (Random)')
        
        ax.set_title(f'{dataset}')
        ax.set_xlabel('Number of Shots')
        ax.set_ylabel('F1 Score')
        
        # Set y-axis limits dynamically
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Set x-axis ticks
        ax.set_xticks(shots)

    # Create a custom legend
    legend_elements = []
    for model in models:
        for encoder in encoders:
            legend_elements.append(plt.Line2D([0], [0], color=encoder_colors[encoder], lw=1, 
                                              label=f'{model} - {encoder} (Embedding)'))
            if encoder == 'clip':
                legend_elements.append(plt.Line2D([0], [0], color=encoder_colors[encoder], lw=1, 
                                                  linestyle=':', label=f'{model} - {encoder} (Random)'))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=len(models) * (len(encoders) + 1))

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.3)  # Adjusted top margin and hspace

    # Save the figure
    plt.savefig(os.path.join(plots_folder, 'identification_plots.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Plots saved in '{plots_folder}/identification_plots.pdf'")

# Main execution
if __name__ == "__main__":
    analysis_folder = 'results_analysis'

    # Load the result_table_dict from the pickle file
    with open(os.path.join(analysis_folder, 'result_table_dict.pkl'), 'rb') as f:
        result_table_dict = pickle.load(f)

    generate_plots(result_table_dict)