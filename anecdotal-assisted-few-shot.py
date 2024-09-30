import pandas as pd
import os

def generate_latex_output(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col=0)
    row = df.iloc[0]
    
    image_path = row['0']
    ground_truth = row['1']
    
    # VIT Embedding examples
    vit_embedding_shots = [path.replace('./', '') for path in row['Embedding Example Paths 4'].strip("[]").replace("'", "").split(', ')]
    vit_embedding_categories = row['Embedding Example Categories 4'].strip("[]").replace("'", "").split(', ')
    
    # Random examples
    random_shots = [path.replace('./', '') for path in row['Random Example Paths 4'].strip("[]").replace("'", "").split(', ')]
    random_categories = row['Random Example Categories 4'].strip("[]").replace("'", "").split(', ')
    
    latex_output = f"""
    \\begin{{figure}}[t!]
        \\small
        \\textbf{{Question: What type of bean leaf lesion is shown in this image?}}
        \\vspace{{1em}}
        \\begin{{center}}
        \\includegraphics[height=0.20\\linewidth]{{{image_path}}}
        \\end{{center}}
        \\vspace{{1em}}
        
        \\begin{{tabular}}{{|p{{0.28\\linewidth}}|p{{0.28\\linewidth}}|p{{0.28\\linewidth}}|}}
            \\hline
            \\textbf{{Category}} & \\textbf{{Subcategory}} & \\textbf{{Task}} \\\\
            \\hline
            Plant Pathology & Bean Leaf Lesions & Classification \\\\
            \\hline
        \\end{{tabular}}
        \\vspace{{1em}}

        \\textbf{{Ground Truth:}} {ground_truth}
        \\vspace{{1em}}

        \\begin{{tabular}}{{p{{0.15\\linewidth}}|p{{0.2\\linewidth}}p{{0.2\\linewidth}}p{{0.2\\linewidth}}p{{0.2\\linewidth}}}}
            \\textbf{{Method}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Examples}}}} \\\\
            \\hline
    """
    
    # VIT Embedding row
    latex_output += "\\textbf{VIT Embedding} & "
    for i in range(4):
        vit_path = os.path.join("/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib", vit_embedding_shots[i])
        latex_output += f"""
            \\begin{{tabular}}{{c}}
            \\includegraphics[width=0.85\\linewidth]{{{vit_path}}} \\\\
            {vit_embedding_categories[i]}
            \\end{{tabular}} &"""
    latex_output = latex_output.rstrip("&") + "\\\\\n"

    # Random row
    latex_output += "\\textbf{Random} & "
    for i in range(4):
        random_path = os.path.join("/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib", random_shots[i])
        latex_output += f"""
            \\begin{{tabular}}{{c}}
            \\includegraphics[width=0.85\\linewidth]{{{random_path}}} \\\\
            {random_categories[i]}
            \\end{{tabular}} &"""
    latex_output = latex_output.rstrip("&") + "\\\\\n"

    latex_output += """
        \\end{tabular}
    \\end{figure}
    """
    
    return latex_output

csv_file_path = 'results/GPT-4o/vit/Bean Leaf Lesions.csv'
latex_output = generate_latex_output(csv_file_path)
print(latex_output)

with open('anecdotal-assisted-few-shot.tex', 'w') as f:
    f.write(latex_output)