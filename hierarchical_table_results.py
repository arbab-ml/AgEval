import pandas as pd
import pickle
import numpy as np

TAXONOMIC_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
SHOTS = [1, 8]

# Load the hierarchical result table dict
with open('results-hierarchical-analysis/hierarchical_result_table_dict.pkl', 'rb') as f:
    result_table_dict = pickle.load(f)

def prepare_hierarchical_data(result_table_dict, shots):
    # Create a DataFrame for each taxonomic level
    rows = []
    
    for level in TAXONOMIC_LEVELS:
        level_results = result_table_dict[shots][level]
        
        # Calculate average values directly from the DataFrame
        # Instead of trying to access a pre-computed 'Average' row
        baseline = level_results['F1']['Random']['vit'].mean()
        stage_val = level_results['F1']['Embedding']['vit'].mean()
        same_examples = level_results['Avg_Same_Category']['Embedding']['vit'].mean()
        
        delta = stage_val - baseline
        
        row = {
            'Level': level.capitalize(),
            'Baseline': f"{baseline:.2f}",
            'STAGE': f"{stage_val:.2f} ({'+' if delta >= 0 else ''}{delta:.2f})",
            'Same Examples': f"{same_examples:.2f}"
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = ['Level', 'Baseline', 'STAGE', 'Same Examples']
    df = df[column_order]
    
    return df

def to_latex_with_multicolumn(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_highlight = df.copy()
    
    # Function to extract the first number from a string
    def extract_number(s):
        try:
            return float(s.split()[0])
        except:
            return float(s)
    
    # Find the highest value in each row (excluding Level and Same Examples columns)
    for idx, row in df_highlight.iterrows():
        baseline = extract_number(row['Baseline'])
        stage = extract_number(row['STAGE'])
        if stage > baseline:
            df_highlight.at[idx, 'STAGE'] = f"\\colorbox{{yellow!25}}{{{row['STAGE']}}}"
        elif baseline > stage:
            df_highlight.at[idx, 'Baseline'] = f"\\colorbox{{yellow!25}}{{{row['Baseline']}}}"
    
    latex = df_highlight.to_latex(index=False, multicolumn=True, multicolumn_format='c',
                                column_format='l' + 'c'*3,
                                escape=False)
    
    # Replace the default rules with custom ones and add booktabs styling
    latex = latex.replace('\\toprule', '\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline')
    
    return latex

# Prepare tables for both 1-shot and 8-shot
tables = {}
for shots in SHOTS:
    df = prepare_hierarchical_data(result_table_dict, shots)
    latex_table = to_latex_with_multicolumn(df)
    
    # Add caption and label with detailed descriptions
    caption = f"""Performance comparison of STAGE across taxonomic levels using {shots}-shot learning. 
    The table shows F1 scores for both baseline (random selection) and STAGE (similarity-guided selection) approaches, 
    with relative improvements shown in parentheses. The 'Same Examples' column indicates the percentage of selected examples 
    that share the same taxonomic classification as the query image. Higher values indicate better example selection quality. 
    Best performance for each level is highlighted."""

    full_latex_table = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:hierarchical_results_{shots}shot}}
{latex_table}\\end{{table*}}"""
    
    tables[shots] = full_latex_table

# Combine tables with a small vertical space between them
combined_tables = "\n\\vspace{2em}\n".join(tables.values())

# Save the LaTeX tables to a file
with open('hierarchical_tables.tex', 'w') as f:
    f.write(combined_tables)

print("LaTeX tables have been saved to 'hierarchical_tables.tex'") 