# Hierarchical Assisted Few-Shot Learning
This project introduces a novel approach combining hierarchical classification with assisted few-shot learning for vision-language models. The key concepts are:

## Evaluation Approaches

1. **Zero-Shot Learning (Baseline)**
   - ✓ Implemented as Random Few-Shot with 0 examples
   - ✓ Direct prediction without any examples
   - ✓ Serves as absolute baseline

2. **Random Few-Shot Learning with Hierarchical Prediction (This was an old paper)**
   - ✓ Random example selection from entire dataset
   - ✓ No hierarchical constraints in example selection
   - ✓ Still performs prediction at each taxonomic level
   - ✓ Implementation complete
   - Note: Examples don't respect hierarchy but predictions follow taxonomic levels

3. **Hierarchical Assisted Few-Shot Learning (This current paper / work)**
   - ✓ Visually similar examples
   - ✓ Examples respect current taxonomic level
   - ✓ Hierarchical prediction with constrained search space
   - ✓ Core implementation complete

## Current Implementation Status

### Completed Features
- ✓ Zero-shot via random few-shot with 0 examples
- ✓ Random few-shot with hierarchical prediction
- ✓ Vision Transformer (ViT) embedding computation
- ✓ Similarity-based example selection
- ✓ Hierarchical prediction flow
- ✓ Basic accuracy metrics per level
- ✓ Example selection logging
- ✓ Prediction distribution visualization
