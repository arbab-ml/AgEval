import pandas as pd
import ast
import os
import argparse

def read_evaluated_row(file_path, row_index=0):
    df = pd.read_csv(file_path)
    evaluated_rows = df[df['evaluated'] == True]
    if len(evaluated_rows) == 0:
        print("No evaluated rows found!")
        return None
    if row_index >= len(evaluated_rows):
        print(f"Row index {row_index} is out of range. Total evaluated rows: {len(evaluated_rows)}")
        return None
    return evaluated_rows.iloc[row_index]

def extract_hierarchy(row):
    return ast.literal_eval(row['hierarchy'])

def format_examples_for_level(row, level, method='Embedding', shot_count='8'):
    try:
        paths_key = f'{method} Example Paths {level} {shot_count}'
        categories_key = f'{method} Example Categories {level} {shot_count}'
        
        if paths_key not in row or categories_key not in row:
            print(f"Missing keys for {method} {level}")
            return []
            
        paths = row[paths_key]
        categories = row[categories_key]
        
        if pd.isna(paths) or pd.isna(categories):
            print(f"NaN values for {method} {level}")
            return []
            
        paths = ast.literal_eval(paths)
        categories = ast.literal_eval(categories)
        
        # Print for debugging
        print(f"\nProcessing {method} examples for {level}:")
        print(f"Paths: {paths[:4]}")
        print(f"Categories: {categories[:4]}")
        
        return list(zip(paths[:4], categories[:4]))
    except Exception as e:
        print(f"Error processing {method} examples for {level}: {str(e)}")
        return []

def get_prediction(row, level, method='Embedding', shot_count='8'):
    pred = row[f'{method} {level} {shot_count}']
    return pred.replace('_', '\\_') if isinstance(pred, str) else pred

def generate_latex(row_index=0):
    workspace_path = "/Users/muhammadarbabarshad/AgEval"
    csv_file = os.path.join(workspace_path, "results-hierarchical/GPT-4o-mini/vit/BioTrove-Balanced_219species_10samples_eval0pct.csv")
    row = read_evaluated_row(csv_file, row_index)
    if row is None:
        return
        
    print(f"Processing evaluated row {row_index} with image: {row['0']}")
    hierarchy = extract_hierarchy(row)
    
    latex_code = r"""
\begin{figure}[p]
    \centering
    \begin{minipage}{0.95\textwidth}
        \centering
        \fbox{
            \begin{minipage}{0.15\textwidth}
                \centering
                \textbf{\small Input Image}\\[2mm]
                \includegraphics[width=0.95\textwidth]{%s}\\[1mm]
                \small Query: Identify taxonomic classification
            \end{minipage}
        }
    \end{minipage}\vspace{2mm}
    """ % (os.path.join(workspace_path, row['0']))
    
    # Add table header with rotated prediction columns
    latex_code += r"""
    \begin{minipage}{0.95\textwidth}
        \centering
        \small
        \begin{tabular}{|l|cccc|c||cccc|c|}
        \hline
        \multirow{2}{*}{\scriptsize Level} & \multicolumn{5}{c||}{\scriptsize STAGE} & \multicolumn{5}{c|}{\scriptsize Traditional} \\
        \cline{2-11}
        & \multicolumn{4}{c|}{\scriptsize Examples} & \rotatebox{90}{\scriptsize Pred.} & \multicolumn{4}{c|}{\scriptsize Examples} & \rotatebox{90}{\scriptsize Pred.} \\
        \hline
    """
    
    for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
        stage_examples = format_examples_for_level(row, level, 'Embedding', '8')
        random_examples = format_examples_for_level(row, level, 'Random', '8')
        stage_pred = get_prediction(row, level, 'Embedding', '8')
        random_pred = get_prediction(row, level, 'Random', '8')
        
        if not stage_examples or not random_examples:
            print(f"Skipping {level} due to missing examples")
            continue
            
        latex_code += r"""
        \rotatebox{90}{\parbox{1.5cm}{\centering\scriptsize \textbf{%s:}\\\scriptsize %s}} & 
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \rotatebox{90}{\scriptsize %s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \includegraphics[width=0.06\textwidth,height=0.03\textheight]{%s} &
        \rotatebox{90}{\scriptsize %s} \\
        \scriptsize & \scriptsize %s & \scriptsize %s & \scriptsize %s & \scriptsize %s & &
        \scriptsize %s & \scriptsize %s & \scriptsize %s & \scriptsize %s & \\
        \hline""" % (
            level.upper(),
            hierarchy[level],
            os.path.join(workspace_path, stage_examples[0][0]),
            os.path.join(workspace_path, stage_examples[1][0]),
            os.path.join(workspace_path, stage_examples[2][0]),
            os.path.join(workspace_path, stage_examples[3][0]),
            stage_pred,
            os.path.join(workspace_path, random_examples[0][0]),
            os.path.join(workspace_path, random_examples[1][0]),
            os.path.join(workspace_path, random_examples[2][0]),
            os.path.join(workspace_path, random_examples[3][0]),
            random_pred,
            stage_examples[0][1], stage_examples[1][1], stage_examples[2][1], stage_examples[3][1],
            random_examples[0][1], random_examples[1][1], random_examples[2][1], random_examples[3][1]
        )
    
    latex_code += r"""
        \end{tabular}
    \end{minipage}
    \caption{Hierarchical analysis showing example selection and predictions at each taxonomic level. For each level, we compare STAGE (our method) with Traditional (random) selection.}
    \label{fig:hierarchical_analysis}
\end{figure}
"""
    
    # Print to console
    print(latex_code)
    
    # Save to file
    output_dir = os.path.join(workspace_path, "writing/677da2e667bc14b45df7d4ae")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "justfigure.tex")
    
    with open(output_file, 'w') as f:
        f.write(latex_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--row_index", type=int, default=4, help="Index of the evaluated row to visualize (0-based)")
    args = parser.parse_args()
    generate_latex(args.row_index) 