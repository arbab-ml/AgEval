import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def read_results(file_path):
    df = pd.read_csv(file_path)
    return df[df['evaluated'] == True]

def extract_hierarchy(row):
    return ast.literal_eval(row['hierarchy'])

def format_examples(row, level, method='Embedding'):
    paths = row[f'{method} Example Paths {level} 1']
    categories = row[f'{method} Example Categories {level} 1']
    if pd.isna(paths) or pd.isna(categories):
        return []
    paths = ast.literal_eval(paths)
    categories = ast.literal_eval(categories)
    return list(zip(paths, categories))

def visualize_examples(csv_file):
    df = read_results(csv_file)
    
    for idx, row in df.iterrows():
        print("\n" + "="*80)
        print(f"Input Image: {row[0]}")
        print(f"Species: {row[1]}")
        print("="*80)
        
        hierarchy = extract_hierarchy(row)
        
        for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            print(f"\n{level.upper()}: {hierarchy[level]}")
            
            # Embedding-based examples
            emb_examples = format_examples(row, level)
            if emb_examples:
                print("\nEmbedding-based examples:")
                for path, category in emb_examples:
                    print(f"  - {os.path.basename(path)} ({category})")
            
            # Random examples
            rand_examples = format_examples(row, level, 'Random')
            if rand_examples:
                print("\nRandom examples:")
                for path, category in rand_examples:
                    print(f"  - {os.path.basename(path)} ({category})")
            
            print("-"*40)

if __name__ == "__main__":
    csv_file = "results-hierarchical/GPT-4o-mini/vit/BioTrove-Balanced_219species_10samples_eval0pct.csv"
    visualize_examples(csv_file) 