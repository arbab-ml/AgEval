import pandas as pd
import pickle
import numpy as np

# Load the result_table_dict
with open('results_analysis/result_table_dict.pkl', 'rb') as f:
    result_table_dict = pickle.load(f)

def prepare_data(result_table_dict):
    baseline = result_table_dict[8]['Random', 'clip']
    eight_shot = result_table_dict[8]['Embedding']
    
    df = pd.DataFrame({
        'Task': [dataset[1] for dataset in baseline.index],
        'Baseline': baseline,
    })
    
    # Calculate improvements and format the output
    for col in ['clip', 'resnet', 'vit']:
        df[col] = eight_shot[col].apply(lambda x: f"{x:.2f}")
        delta = eight_shot[col] - df['Baseline']
        df[col] += delta.apply(lambda x: f" ({'+' if x >= 0 else ''}{x:.2f})")
    
    # Format baseline column
    df['Baseline'] = df['Baseline'].apply(lambda x: f"{x:.2f}")
    
    # Reorder columns
    column_order = ['Task', 'Baseline', 'clip', 'resnet', 'vit']
    df = df[column_order]
    
    return df

def to_latex_with_multicolumn(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_highlight = df.copy()
    
    # Function to extract the first number from a string
    def extract_number(s):
        return float(s.split()[0])
    
    # Find the highest value in each row (excluding 'Task' column)
    for idx, row in df_highlight.iloc[:, 1:].iterrows():
        max_val = max(extract_number(val) for val in row)
        for col in df_highlight.columns[1:]:
            if extract_number(df_highlight.at[idx, col]) == max_val:
                df_highlight.at[idx, col] = f"\\colorbox{{yellow!25}}{{{df_highlight.at[idx, col]}}}"
    
    latex = df_highlight.to_latex(index=False, multicolumn=True, multicolumn_format='c',
                                  column_format='l' + 'c'*4, 
                                  escape=False)
    
    # Remove the \toprule, \midrule, and \bottomrule commands
    latex = latex.replace('\\toprule', '\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline')
    
    return latex

# Prepare the data
df = prepare_data(result_table_dict)

# Convert to LaTeX
latex_table = to_latex_with_multicolumn(df)

# Add caption and label
full_latex_table = r"""\begin{table*}[htbp]
\centering
\caption{Assisted Few-shot Performance Comparison with Baseline (8-shot)}
\label{tab:eight_shot_results}
\resizebox{\textwidth}{!}{%
""" + latex_table + r"}\end{table*}"

# Save the LaTeX table to a file
with open('table.tex', 'w') as f:
    f.write(full_latex_table)

print("LaTeX table has been saved to 'table.tex'")