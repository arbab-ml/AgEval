import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import plotly.express as px
import colorsys
from typing import Dict

# Configuration for representative sampling
SAMPLING_CONFIG = {
    'class': {'top_n': 6, 'min_samples': 50},    # Show top 6 classes with at least 50 samples
    'order': {'top_n': 3, 'min_samples': 30},     # Show top 3 orders per class with at least 30 samples
    'family': {'top_n': 3, 'min_samples': 20},    # Show top 3 families per order with at least 20 samples
    'genus': {'top_n': 2, 'min_samples': 10},     # Show top 2 genera per family with at least 10 samples
    'species': {'top_n': 2, 'min_samples': 5}     # Show top 2 species per genus with at least 5 samples
}

def get_color_shades(base_color, num_shades):
    """Generate shades of a base color."""
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    rgb = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to HSV
    hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
    colors = []
    for i in range(num_shades):
        # Vary saturation and value, keep hue constant
        sat = min(1.0, hsv[1] + (i * 0.1))  # Increase saturation
        val = max(0.3, hsv[2] - (i * 0.1))  # Decrease value
        
        # Convert back to RGB
        rgb = colorsys.hsv_to_rgb(hsv[0], sat, val)
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def filter_representative_nodes(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter the dataset to show only representative nodes based on configuration.
    """
    df = df.copy()
    
    # Start with class level (depth 2)
    df['depth'] = df['id'].str.count('/')
    df_filtered = df[df['depth'] == 2].copy()
    
    # Sort by value and take top N classes
    df_filtered = df_filtered.nlargest(
        n=config['class']['top_n'], 
        columns='value'
    )
    df_filtered = df_filtered[df_filtered['value'] >= config['class']['top_n']]
    
    # For each selected class, process its hierarchy
    selected_ids = set(df_filtered['id'])
    taxonomic_levels = ['order', 'family', 'genus', 'species']
    
    for level in taxonomic_levels:
        current_depth = taxonomic_levels.index(level) + 3  # Depth starts at 3 for orders
        level_df = df[df['depth'] == current_depth].copy()
        
        # For each parent in the previous level
        new_selected = set()
        for parent_id in selected_ids:
            # Get children of this parent
            children = level_df[level_df['parent'] == parent_id]
            if not children.empty:
                # Select top N children with minimum samples
                top_children = children.nlargest(
                    n=config[level]['top_n'], 
                    columns='value'
                )
                top_children = top_children[
                    top_children['value'] >= config[level]['min_samples']
                ]
                new_selected.update(top_children['id'])
        
        # Add selected nodes for this level
        selected_ids.update(new_selected)
    
    # Return filtered dataframe with only selected nodes and their connections
    return df[df['id'].isin(selected_ids)]

def create_taxonomy_sunburst(sampling_config: Dict = SAMPLING_CONFIG):
    """
    Create a sunburst visualization of the taxonomy structure.
    """
    # Load the prepared data
    df = pd.read_csv('taxonomy_data/sunburst_data.csv')
    
    # Filter to representative nodes
    df_filtered = filter_representative_nodes(df, sampling_config)
    
    # Remap parents for class level to empty string (making them roots)
    df_filtered.loc[df_filtered['depth'] == 2, 'parent'] = ''
    
    # Base colors for classes (using a color-blind friendly palette)
    class_colors = {
        'Insecta': '#4575b4',      # Blue
        'Aves': '#74add1',         # Light Blue
        'Mammalia': '#313695',     # Dark Blue
        'Bryopsida': '#1a9850',    # Green
        'Liliopsida': '#66bd63',   # Light Green
        'Polypodiopsida': '#006837',# Dark Green
        'Jungermanniopsida': '#d73027',  # Red
        'Anthocerotopsida': '#fc8d59',   # Orange
        'Other': '#969696'         # Gray
    }
    
    # Create a color mapping for all nodes
    color_map = {}
    
    for idx, row in df_filtered.iterrows():
        path_parts = row['id'].split('/')
        depth = len(path_parts) - 1
        
        if depth == 2:  # Class level
            color_map[row['id']] = class_colors.get(row['label'], class_colors['Other'])
        else:
            # Get parent's color and create a shade
            parent_id = row['parent']
            parent_color = color_map.get(parent_id)
            if parent_color:
                shades = get_color_shades(parent_color, 5)  # 5 levels (class to species)
                color_map[row['id']] = shades[depth-3]  # Adjust depth offset
            else:
                color_map[row['id']] = class_colors['Other']
    
    # Create the sunburst plot
    fig = go.Figure(go.Sunburst(
        ids=df_filtered['id'],
        parents=df_filtered['parent'],
        labels=df_filtered['label'],
        values=df_filtered['value'],
        marker=dict(
            colors=[color_map[id_] for id_ in df_filtered['id']],
            line=dict(color='white', width=0.5)
        ),
        branchvalues='total',
        maxdepth=5,  # Show 5 levels (class to species)
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Samples: %{value}<extra></extra>'
    ))
    
    # Add class color samples to legend
    for class_name, color in class_colors.items():
        if class_name != 'Other':  # Skip 'Other' in legend
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=class_name,
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Taxonomic Structure by Class',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        width=1200,
        height=1200,
        showlegend=True,
        legend=dict(
            title=dict(text='Classes'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        ),
        margin=dict(t=100, l=20, r=150, b=20)
    )
    
    return fig

def main():
    # Example of using different configurations
    configs = {
        'detailed': {
            'class': {'top_n': 6, 'min_samples': 50},
            'order': {'top_n': 3, 'min_samples': 30},
            'family': {'top_n': 3, 'min_samples': 20},
            'genus': {'top_n': 2, 'min_samples': 10},
            'species': {'top_n': 2, 'min_samples': 5}
        },
        'minimal': {
            'class': {'top_n': 4, 'min_samples': 100},
            'order': {'top_n': 2, 'min_samples': 50},
            'family': {'top_n': 2, 'min_samples': 30},
            'genus': {'top_n': 2, 'min_samples': 20},
            'species': {'top_n': 1, 'min_samples': 10}
        }
    }
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations with different configurations
    for config_name, config in configs.items():
        fig = create_taxonomy_sunburst(config)
        
        # Save as interactive HTML
        fig.write_html(output_dir / f'taxonomy_structure_{config_name}.html')
        
        # Save as high-resolution PNG for publication
        fig.write_image(
            output_dir / f'taxonomy_structure_{config_name}.png',
            width=2400, height=2400, scale=2
        )

if __name__ == "__main__":
    main() 