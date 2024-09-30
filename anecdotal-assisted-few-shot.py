import pandas as pd
import os


#Metadata about all # DO NOT DELETE THESE:
dataset_mapping = {
    'Durum Wheat': ('Identification (I)', 'F1', 'Seed Morphology'),
    'Soybean Seeds': ('Identification (I)', 'F1', 'Seed Morphology'),
    'Mango Leaf Disease': ('Identification (I)', 'F1', 'Foliar Stress'),
    'Bean Leaf Lesions': ('Identification (I)', 'F1', 'Foliar Stress'),
    'Soybean Diseases': ('Identification (I)', 'F1', 'Foliar Stress'),
    'Dangerous Insects': ('Identification (I)', 'F1', 'Invasive Species'),
    'DeepWeeds': ('Identification (I)', 'F1', 'Invasive Species'),
    'Yellow Rust 19': ('Classification (C)', 'NMAE', 'Disease Severity'),
    'IDC': ('Classification (C)', 'NMAE', 'Stress Tolerance'),
    'FUSARIUM 22': ('Classification (C)', 'NMAE', 'Stress Tolerance'),
    'InsectCount': ('Quantification (Q)', 'NMAE', 'Pest'),
    'PlantDoc': ('Quantification (Q)', 'NMAE', 'Disease'),
}

question_mapping = {
    'Durum Wheat':'What wheat variety is this?',
    'Soybean Seeds':'What soybean lifecycle stage is this?' ,
    'Mango Leaf Disease': 'What mango leaf disease is present?' ,
    'Bean Leaf Lesions': 'What type of bean leaf lesion is this?',
    'Soybean Diseases': 'What is the type of stress in this soybean?',
    'Dangerous Insects': 'What is the name of this harmful insect?',
    'DeepWeeds':'What is the name of this weed?' ,
    'Yellow Rust 19':'What is the severity of yellow rust disease?' ,
    'IDC':'What is the rating (1-5) of soybean stress severity?' ,
    'FUSARIUM 22':'What is the severity of chickpea fusarium wilt?' ,
    'InsectCount': 'What is the insect count?',
    'PlantDoc': 'What is the diseased leaf percentage?',
}

# THE NAMES OF FILES ARE:
# results/GPT-4o/vit/Bean Leaf Lesions.csv
# results/GPT-4o/vit/Dangerous Insects.csv
# results/GPT-4o/vit/DeepWeeds.csv
# results/GPT-4o/vit/Durum Wheat.csv
# results/GPT-4o/vit/Mango Leaf Disease.csv
# results/GPT-4o/vit/SBRD.csv
# results/GPT-4o/vit/Soybean Diseases.csv
# results/GPT-4o/vit/Soybean Seeds.csv




def generate_latex_output(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col=0)
    row = df.iloc[0]
    
    dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    prefix = "/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib"
    image_path = os.path.join(prefix, row['0'].replace('./', ''))
    ground_truth = row['1']
    
    question = question_mapping.get(dataset_name, "Question not found")
    category, metric, subcategory = dataset_mapping.get(dataset_name, ("", "", ""))
    task = dataset_name  # The task is the dataset name itself
    
    vit_embedding_shots = [path.replace('./', '') for path in row['Embedding Example Paths 4'].strip("[]").replace("'", "").split(', ')]
    vit_embedding_categories = row['Embedding Example Categories 4'].strip("[]").replace("'", "").split(', ')
    
    random_shots = [path.replace('./', '') for path in row['Random Example Paths 4'].strip("[]").replace("'", "").split(', ')]
    random_categories = row['Random Example Categories 4'].strip("[]").replace("'", "").split(', ')
    
    latex_output = f"""
    \\begin{{figure}}[htbp]
    \\centering
    \\begin{{subfigure}}[c]{{0.25\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{image_path}}}
        \\caption{{Sample image}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[c]{{0.65\\textwidth}}
        \\centering
        \\begin{{tabular}}{{lll}}
            \\toprule
            \\textbf{{Category}} & \\textbf{{Subcategory}} & \\textbf{{Task}} \\\\
            \\midrule
            {category} & {subcategory} & {task} \\\\
            \\midrule
            \\multicolumn{{3}}{{l}}{{\\textbf{{Question:}} {question}}} \\\\
            \\multicolumn{{3}}{{l}}{{\\textbf{{Ground Truth:}} {ground_truth}}} \\\\
            \\bottomrule
        \\end{{tabular}}
    \\end{{subfigure}}

    \\vspace{{1em}}

    \\begin{{tabular}}{{lcccc}}
        \\toprule
        \\textbf{{Method}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Examples}}}} \\\\
        \\midrule
        VIT-based
    """
    
    for i in range(4):
        vit_path = os.path.join(prefix, vit_embedding_shots[i])
        latex_output += f" & \\begin{{tabular}}{{c}}\n"
        latex_output += f"    \\includegraphics[width=0.15\\textwidth]{{{vit_path}}} \\\\\n"
        latex_output += f"    {vit_embedding_categories[i]}\n"
        latex_output += f"\\end{{tabular}}"
    latex_output += " \\\\\n"

    latex_output += "Traditional"
    for i in range(4):
        random_path = os.path.join(prefix, random_shots[i])
        latex_output += f" & \\begin{{tabular}}{{c}}\n"
        latex_output += f"    \\includegraphics[width=0.15\\textwidth]{{{random_path}}} \\\\\n"
        latex_output += f"    {random_categories[i]}\n"
        latex_output += f"\\end{{tabular}}"
    latex_output += " \\\\\n"

    latex_output += """
        \\bottomrule
    \\end{tabular}
    \\caption{Analysis of the """ + f"{dataset_name}" + """ dataset}
    \\label{fig:""" + f"{dataset_name.lower().replace(' ', '_')}" + """}
    \\end{figure}
    """
    
    return latex_output

# csv_file_path = 'results/GPT-4o/vit/Bean Leaf Lesions.csv'
csv_file_path = 'results/GPT-4o/vit/DeepWeeds.csv'

latex_output = generate_latex_output(csv_file_path)
print(latex_output)

with open('anecdotal-assisted-few-shot.tex', 'w') as f:
    f.write(latex_output)