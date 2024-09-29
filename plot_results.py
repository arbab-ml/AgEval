import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

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

    shots = [0, 1, 2, 4, 8]
    datasets = result_table_dict[0].index.get_level_values('Dataset').unique()
    models = result_table_dict[0].index.get_level_values('Model').unique()
    encoders = result_table_dict[0].columns.get_level_values('Encoder').unique()

    n_datasets = len(datasets)
    n_cols = 4
    n_rows = math.ceil(n_datasets / n_cols)

    metrics = ['F1', 'Avg_Same_Category']
    for metric in metrics:
        fig = plt.figure(figsize=(15, 3.2 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4)

        encoder_colors = {'clip': '#1f77b4', 'resnet': '#ff7f0e', 'vit': '#2ca02c'}

        for i, dataset in enumerate(datasets):
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            all_values = []
            for model in models:
                for encoder in encoders:
                    embedding_values = [result_table_dict[shot].loc[(model, dataset), (metric, 'Embedding', encoder)] for shot in shots]
                    all_values.extend(embedding_values)
                    ax.plot(shots, embedding_values, marker='o', markersize=3, linewidth=1, 
                            linestyle='-', color=encoder_colors[encoder], 
                            label=f'{model} - {encoder} (Embedding)')
                    
                    if encoder == 'clip':
                        random_values = [result_table_dict[shot].loc[(model, dataset), (metric, 'Random', encoder)] for shot in shots]
                        all_values.extend(random_values)
                        ax.plot(shots, random_values, marker='o', markersize=3, linewidth=1, 
                                linestyle=':', color=encoder_colors[encoder], 
                                label=f'{model} - {encoder} (Random)')
            
            ax.set_title(f'{dataset}')
            ax.set_xlabel('Number of Shots')
            ax.set_ylabel(metric)
            
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks(shots)

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

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.3)

        plt.savefig(os.path.join(plots_folder, f'{metric.lower()}_plots.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Plots saved in '{plots_folder}/{metric.lower()}_plots.pdf'")

    # Add caption
    caption = "Adaptive Few-Shot Learning for Image Classification: Enhancing model performance by providing smart few-shot examples."
    with open(os.path.join(plots_folder, 'caption.txt'), 'w') as f:
        f.write(caption)

def to_latex_with_multicolumn(df):
    df = df.reset_index()
    
    df.columns = pd.MultiIndex.from_tuples([('Model', ''), ('Dataset', '')] + 
                                           [col for col in df.columns[2:]])
    
    latex = df.to_latex(multicolumn=True, multicolumn_format='c', multirow=True,
                        column_format='l' + 'l' + 'c'*len(df.columns[2:]), 
                        float_format="{:0.2f}".format,
                        escape=False)
    
    latex = latex.replace('\\toprule', '\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline')
    
    return latex

if __name__ == "__main__":
    analysis_folder = 'results_analysis'

    with open(os.path.join(analysis_folder, 'result_table_dict.pkl'), 'rb') as f:
        result_table_dict = pickle.load(f)
    
    generate_plots(result_table_dict)

# %%
