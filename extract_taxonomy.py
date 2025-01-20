import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def extract_taxonomy_structure():
    """
    Extract hierarchical taxonomy structure from the dataset.
    Returns a nested dictionary representing the taxonomy with counts.
    """
    # Load the balanced dataset metadata
    df = pd.read_csv('biotrove-data/balanced_metadata.csv')
    
    # Initialize the taxonomy tree with counts
    taxonomy = {}
    
    # Process each row to build the hierarchical structure
    for _, row in df.iterrows():
        current_level = taxonomy
        
        # Navigate through each taxonomic level
        for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            taxon = row[level]
            
            # Initialize or update node
            if taxon not in current_level:
                current_level[taxon] = {
                    'count': 1,
                    'children': {}
                }
            else:
                current_level[taxon]['count'] += 1
            
            current_level = current_level[taxon]['children']
    
    # Print statistics about the dataset
    print("\nDataset Statistics:")
    def count_unique_at_level(node, level, current_level=0):
        if current_level == level:
            return len(node)
        
        unique_count = 0
        for child in node.values():
            unique_count += count_unique_at_level(child['children'], level, current_level + 1)
        return unique_count
    
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for i, level in enumerate(taxonomic_levels):
        unique_count = count_unique_at_level(taxonomy, i)
        print(f"Unique {level}s: {unique_count}")
    
    return taxonomy

def prepare_sunburst_data(taxonomy):
    """
    Convert taxonomy dictionary to format needed for sunburst plot.
    Returns lists of ids, parents, labels, and values.
    """
    ids = []
    parents = []
    labels = []
    values = []
    
    def process_level(node, parent_id=None, level=0, path=""):
        items = [(k, v) for k, v in node.items()]
        # Sort by count (descending)
        items.sort(key=lambda x: x[1]['count'], reverse=True)
        
        for taxon, data in items:
            # Create unique ID using path to ensure uniqueness
            current_id = f"{path}/{taxon}" if path else taxon
            
            # Add to our lists
            ids.append(current_id)
            parents.append(parent_id if parent_id else "")
            labels.append(taxon)
            values.append(data['count'])
            
            # Process children
            process_level(data['children'], current_id, level + 1, current_id)
    
    process_level(taxonomy)
    
    # Print validation statistics
    print("\nVisualization Statistics:")
    taxonomic_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    level_counts = defaultdict(int)
    
    for id_val, value in zip(ids, values):
        level = len(id_val.split('/')) - 1  # Count depth based on path
        if level < len(taxonomic_levels):
            level_counts[taxonomic_levels[level]] = max(level_counts[taxonomic_levels[level]], value)
    
    for level in taxonomic_levels:
        print(f"{level}: {level_counts[level]} samples")
    
    return ids, parents, labels, values

def save_taxonomy_data():
    """
    Extract taxonomy and save both raw structure and processed data.
    """
    # Extract taxonomy
    taxonomy = extract_taxonomy_structure()
    
    # Save raw taxonomy structure
    output_dir = Path('taxonomy_data')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'taxonomy_structure.json', 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    # Prepare and save data for visualization
    ids, parents, labels, values = prepare_sunburst_data(taxonomy)
    
    # Save as CSV for easy loading
    pd.DataFrame({
        'id': ids,
        'parent': parents,
        'label': labels,
        'value': values
    }).to_csv(output_dir / 'sunburst_data.csv', index=False)

if __name__ == "__main__":
    save_taxonomy_data() 