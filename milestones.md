# Hierarchical Assisted Few-Shot Learning
This project introduces a novel approach combining hierarchical classification with assisted few-shot learning for vision-language models. The key concepts are:

## Evaluation Approaches

1. **Zero-Shot Learning (Baseline)**
   - ✓ Implemented as Random Few-Shot with 0 examples
   - ✓ Direct prediction without any examples
   - ✓ Serves as absolute baseline

2. **Random Few-Shot Learning with Hierarchical Prediction**
   - ✓ Random example selection from entire dataset
   - ✓ No hierarchical constraints in example selection
   - ✓ Still performs prediction at each taxonomic level
   - ✓ Implementation complete
   - Note: Examples don't respect hierarchy but predictions follow taxonomic levels

3. **Hierarchical Assisted Few-Shot Learning (Our Method)**
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

### In Progress
1. **Evaluation Framework**
   - Add theoretical vs. actual success rate comparison
   - Implement similarity score distribution analysis
   - Add example selection quality metrics
   - Compare hierarchical vs. flat classification accuracy

2. **Metrics Enhancement**
   - Add cumulative accuracy across levels
   - Implement confidence scoring
   - Add error pattern analysis across levels
   - Calculate example relevance statistics

3. **Documentation and Analysis**
   - Add mathematical analysis of random selection limitations
   - Document similarity threshold effects
   - Analyze hierarchical improvement patterns
   - Compare performance across different shot counts

## Implementation Details

### Data Structure
```python
{
  'image_path': str,
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

### Key Metrics to Implement
1. Example Selection Quality
   ```python
   - Similarity score distribution
   - % of examples from same taxonomic group
   - Theoretical vs. actual success rates
   ```

2. Hierarchical Performance
   ```python
   - Per-level accuracy
   - Cumulative accuracy
   - Error propagation analysis
   ```

3. Comparative Analysis
   ```python
   - Zero-shot vs. Random Few-shot
   - Random Few-shot vs. Hierarchical Assisted
   - Performance at different shot counts (1-8)
   ```

## Next Steps
1. Complete the evaluation framework
2. Implement additional metrics
3. Add comprehensive performance analysis
4. Document limitations and future work

The project now focuses on thoroughly evaluating these three approaches, with particular emphasis on demonstrating the effectiveness of hierarchical assisted few-shot learning in scenarios with large numbers of classes. 