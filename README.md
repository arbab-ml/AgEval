# AgEval Benchmark Companion Repository

This repository contains the companion code for the AgEval benchmark datasets, focusing on plant stress identification, classification, and quantification. It includes 12 subsets of data used in the benchmark.

## Overview

The main components of this repository are:

1. `inference.py`: Script for evaluating models on the AgAgEval Benchmark Companion RepositoryEval datasets.
2. `data_loader.py`: Functions for downloading and preparing the benchmark datasets.

To replicate the results presented in the paper, run `inference.py` to evaluate no-context or few-shot in-context learning on the datasets.

## Inference (`inference.py`)

The `inference.py` script contains:

- Implementation of multiple AI models (OpenAI, Anthropic, Google, OpenRouter)
- Functions for asynchronous processing to improve performance
- Progress tracking using tqdm
- Result saving in CSV format
- Evaluation of no-context and few-shot in-context learning
- Customizable number of shots for in-context learning

### Supported Models

1. GPT-4 (OpenAI)
2. Claude-3.5-sonnet (Anthropic)
3. Claude-3-haiku (Anthropic)
4. LLaVA v1.6 34B (OpenRouter)
5. Gemini-flash-1.5 (Google)
6. Gemini-pro-1.5 (Google)

## Data Loader (`data_loader.py`)

The `data_loader.py` script provides functions to download and prepare the 12 AgEval benchmark datasets. Features include:

- Dataset-specific loading functions
- Automatic downloading from Kaggle or Zenodo if not present
- Extraction and renaming of files in the `/data` folder
- Random sampling with a fixed seed for reproducibility

### Available Datasets

1. Durum Wheat Dataset
2. Soybean Seeds Dataset
3. Mango Leaf Disease Dataset
4. DeepWeeds Dataset
5. Bean Leaf Lesions Dataset
6. Yellow Rust 19 Dataset
7. FUSARIUM 22 Dataset
8. PlantDoc (Leaf Disease Segmentation)
9. Dangerous Insects Dataset
10. IDC (Iron Deficiency Chlorosis) Dataset
11. Soybean Diseases Dataset (PNAS)
12. InsectCount Dataset

### Usage

Each dataset loading function accepts a `total_samples_to_check` parameter (default is 100) to specify the number of samples per class for evaluation:

```python
from data_loader import load_and_prepare_data_DurumWheat

samples, classes, dataset_name = load_and_prepare_data_DurumWheat(total_samples_to_check=50)
```

## Notes

- The scripts will skip downloading datasets if they already exist in the `/data` folder.
- Evaluation results are saved in the `/results` folder, organized by model name and dataset.
- For detailed information on each dataset and the evaluation process, please refer to the AgEval benchmark paper.

For more detailed information about the implementation, please refer to the comments in the source code files.
