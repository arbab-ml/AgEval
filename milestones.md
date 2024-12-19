# Current Implementation
- Assisted few-shot learning: provides examples by finding top-k matching images using embedding model similarity
- Single level classification: directly predicts species/class label

# Proposed Implementation: Hierarchical Assisted Few-Shot Learning
The new implementation will maintain backward compatibility while adding hierarchical prediction capabilities.

## Phase 1: Data Structure Updates
1. Update DataLoader Interface
   - Create new hierarchical data loading functions that return both flat and hierarchical labels
   - Modify existing data loaders to optionally return hierarchical information
   - Structure: 
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

2. Metadata Processing
   - Extract hierarchical information from BioTrove metadata.csv
   - Create taxonomic level mappings
   - Handle datasets without complete hierarchical information by using flat labels

## Phase 2: Core Logic Implementation
1. Embedding Similarity Logic
   - Keep existing embedding functions unchanged for backward compatibility
   - Add new function for hierarchical similarity:
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

2. Prediction Flow
   - Implement sequential prediction through taxonomy:
     ```python
     async def hierarchical_predict(image_path, all_data):
         predictions = {}
         current_filter = {}
         
         for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
             # Get examples for current level
             examples = get_hierarchical_examples(image_path, all_data, level)
             
             # Filter examples based on previous predictions
             filtered_examples = filter_by_predictions(examples, current_filter)
             
             # Get prediction for current level
             prediction = await get_level_prediction(image_path, filtered_examples, level)
             
             # Store prediction and update filter
             predictions[level] = prediction
             current_filter[level] = prediction
         
         return predictions
     ```

3. Prompt Engineering
   - Create level-specific prompts:
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

## Phase 3: API Integration
1. Update Process Image Function
   - Add hierarchical_mode parameter (default=False)
   - Modify function signature:
     ```python
     async def process_image(api, i, number_of_shots, all_data_results, all_data, 
                           progress_bar, embeddings, use_embedding=True, 
                           hierarchical_mode=False, encoder="vit"):
     ```

2. Results Storage
   - New columns for hierarchical results:
     - Kingdom_Prediction
     - Phylum_Prediction
     - Class_Prediction
     - Order_Prediction
     - Family_Prediction
     - Genus_Prediction
     - Species_Prediction
   - Maintain existing columns for backward compatibility

## Phase 4: Evaluation Metrics
1. Accuracy Metrics
   - Per-level accuracy calculation
   - Cumulative accuracy (correct predictions up to each level)
   - Comparison with flat classification baseline

2. Results Analysis
   - Generate accuracy metrics for each taxonomic level
   - Compare hierarchical vs flat classification performance
   - Analyze error patterns across taxonomic levels

## Implementation Order
1. Start with BioTrove dataset integration and hierarchical data structure
2. Implement hierarchical example selection
3. Add level-specific prompts and prediction flow
4. Integrate with existing API framework
5. Add results storage and evaluation metrics

## Notes
- Default mode remains backward compatible with current implementation
- Hierarchical mode activated by hierarchical_mode=True parameter
- BioTrove dataset serves as primary test case
- Other datasets can fall back to flat classification if lacking hierarchical information
- Most Relavent files could are: @inference.py @data_loader.py @metadata.csv @get_embeddingp.py @download_biotrove.py @milestones.md