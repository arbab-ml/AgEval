# Good to Know

## Template

### Feature/Component Name
**File:** [filename]  
**Purpose:** Brief description of what this feature/component does  
**How it works:** Short explanation of the implementation  
**Usage:** How to use this feature in the codebase  

---

## Request and Response Logging

**File:** inference.py  
**Purpose:** Tracks and displays progress of API requests and responses during image processing  

**How it works:**  
The system uses a `ProgressBar` class with two tqdm progress bars to visualize:
1. Outgoing API requests ("Requests sent")
2. Incoming API responses ("Responses received")

The class provides thread-safe methods (`update_sent()` and `update_received()`) that increment the respective progress bars. These methods are called before sending an API request and after processing a response.

**Usage:**  
```python
# Initialize progress bar
progress_bar = ProgressBar(total_number_of_requests)

# Before sending API request
await progress_bar.update_sent()

# After processing response
await progress_bar.update_received()

# When all processing is complete
progress_bar.close()
```

The dual-progress bar system provides visual feedback during long-running tasks with many API calls, making it easier to monitor the progress of the image processing pipeline.

---

## Hierarchical Classification Error Handling

**Files:** inference.py, compile_results.py  
**Purpose:** Manages error handling in hierarchical taxonomic classification  

**How it works:**  
The system uses a simplified approach with different behaviors based on the example selection mode:

1. **In inference.py**:
   - **Embedding Mode**: If an error occurs at any taxonomic level, processing stops for that image. Also, if a previous level had an error, the current level is skipped.
   - **Random Mode**: If an error occurs, processing continues to the next taxonomic level, since example selection doesn't depend on previous predictions.
   - No explicit "cascade error" marking in the inference code - errors are simply recorded as they occur.

2. **In compile_results.py**:
   - Cascade errors are identified during analysis by checking if any higher taxonomic level has an error.
   - This post-processing approach simplifies the inference code while still providing accurate error reporting.
   - The analysis distinguishes between direct errors (occurring at the current level) and cascade errors (propagated from higher levels).

**Error Types:**
- `JSON_PARSE`: Failed to parse API response as JSON
- `INVALID_PRED`: Prediction not in valid options
- `API_ERROR`: API call failed
- `GENERAL_ERROR`: Other unexpected errors
- `CASCADE_ERROR`: Identified during analysis when higher levels have errors

This simplified approach keeps the inference code cleaner while still providing detailed error analysis in the results. 