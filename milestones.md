# Hierarchical Assisted Few-Shot Learning
This project introduces a novel approach combining hierarchical classification with assisted few-shot learning for vision-language models. The key concepts are:

## Assisted Few-Shot Learning
- Traditional few-shot learning uses random examples for prompting
- Our approach intelligently selects examples based on visual similarity
- Uses embedding models (ViT, ResNet, CLIP) to compute image similarities
- Selects the most relevant examples for each prediction

## Hierarchical Classification
- Instead of direct species prediction, follows biological taxonomy
- Predicts in sequence: kingdom → phylum → class → order → family → genus → species
- Each level's prediction informs and constrains the next level
- Reduces the classification space at each step
- Leverages the hierarchical structure of biological taxonomy

# Current Implementation Status
## Completed
- Assisted few-shot learning with embedding-based similarity
- Single level classification baseline
- Hierarchical data structure and loading
- Hierarchical prediction flow
- Level-specific prompts
- Random example selection from full dataset
- Basic accuracy metrics per level

## In Progress/Remaining
1. Data Structure Refinements
   - Optimize hierarchical data loading for other datasets
   - Add support for incomplete taxonomic hierarchies

2. Core Logic Enhancements
   - Add cumulative accuracy across levels
   - Implement confidence scoring
   - Add support for partial hierarchies
   - Consider adding weighted similarity metrics

3. Evaluation Metrics
   - Implement confusion matrices per level
   - Add cumulative accuracy (correct predictions up to each level)
   - Add error pattern analysis across levels

## Implementation Details

### Data Structure
```python
{
  'image_path': str,
  'flat_label': str,  # for backward compatibility
  'hierarchy': {
    'kingdom': str,
    'phylum': str,
    'class': str,
    'order': str,
    'family': str,
    'genus': str,
    'species': str
  }
}
```

### Embedding Similarity Logic
```python
def get_hierarchical_examples(query_image, all_data, taxonomic_level, n_examples=5):
    # Get embeddings
    query_embedding = get_image_embedding(query_image)
    
    # Filter examples by current taxonomic level
    level_filtered_data = filter_by_level(all_data, taxonomic_level)
    
    # Get similar examples at current level
    similar_examples = get_top_k_similar(query_embedding, level_filtered_data, k=n_examples)
    return similar_examples
```

### Hierarchical Prompts
```python
hierarchical_prompts = {
    'kingdom': "Identify the kingdom of this organism. Options: {options}",
    'phylum': "For this {kingdom} organism, identify its phylum. Options: {options}",
    'class': "Within the phylum {phylum}, identify the class. Options: {options}",
    'order': "Within the class {class}, identify the order. Options: {options}",
    'family': "Within the order {order}, identify the family. Options: {options}",
    'genus': "Within the family {family}, identify the genus. Options: {options}",
    'species': "Within the genus {genus}, identify the species. Options: {options}"
}
```

## Important Notes
1. Random Selection:
   - Now selects from entire dataset without hierarchical filtering
   - Provides better baseline comparison with embedding-based selection
   - Used to validate the effectiveness of assisted selection

2. Embedding Selection:
   - Maintains hierarchical filtering for context-aware example selection
   - Uses ViT embeddings by default (CLIP and ResNet available)
   - Selects examples based on visual similarity at each taxonomic level

3. Accuracy Calculation:
   - Handles 'NA' values properly
   - Converts all comparisons to string type for consistency
   - Reports per-level accuracy statistics

4. Future Considerations:
   - Consider adding confidence thresholds for predictions
   - Implement early stopping if confidence is too low
   - Add support for partial hierarchies in other datasets
   - Consider adding cross-validation for hyperparameter tuning

## Project Overview
The project enhances few-shot learning in vision-language models through Assisted Few-Shot Learning. It focuses on improving image classification tasks with limited annotated data by leveraging multiple image encoders: Vision Transformer (ViT), ResNet-50, and CLIP encoder.

The proposed method aims to enhance model performance by providing smart few-shot examples through:
1. Analyzing the current input image using embedding models
2. Selecting similar examples from the training set based on embedding similarity
3. Using these similar examples as few-shot prompts at each taxonomic level
4. Evaluating and comparing performance with random few-shot selection

This approach combines two key innovations:
1. Assisted example selection: Using visual similarity to find relevant examples
2. Hierarchical prediction: Following biological taxonomy to break down the classification task

The hypothesis is that similar examples improve model performance more effectively than random ones, especially when combined with hierarchical classification. This is now being validated through proper random baseline selection and hierarchical accuracy metrics.