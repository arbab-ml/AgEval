import requests
import base64
import os
from typing import List, Union, Dict, Any
from PIL import Image
import io
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, ViTModel, ResNetModel, ResNetConfig, AutoFeatureExtractor
import pickle

AVAILABLE_ENCODERS = ["vit"]  # Focusing on ViT for now as it's most reliable for hierarchical classification

# ViT model setup - using a larger variant for better hierarchical feature extraction
VIT_MODEL_NAME = "google/vit-large-patch16-224-in21k"
vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
vit_model = ViTModel.from_pretrained(VIT_MODEL_NAME)

# Global cache for embeddings
EMBEDDING_CACHE = {}
CACHE_FILE = "embedding_cache.pkl"
SAVE_FREQUENCY = 10  # Save cache every N embeddings

def load_cache():
    """Load embedding cache from disk if it exists."""
    global EMBEDDING_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                EMBEDDING_CACHE = pickle.load(f)
            print(f"Loaded {len(EMBEDDING_CACHE)} cached embeddings")
        except Exception as e:
            print(f"Error loading cache: {e}")
            EMBEDDING_CACHE = {}

def save_cache():
    """Save embedding cache to disk."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(EMBEDDING_CACHE, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_image_embedding(image_path: str, model_type: str = "vit") -> Union[List[float], Dict[str, str]]:
    """
    Get image embedding optimized for hierarchical classification.
    Uses caching to avoid recomputing embeddings for the same image.
    
    Args:
        image_path: Path to the image file
        model_type: Type of model to use for embedding (currently only 'vit' supported)
    
    Returns:
        Image embedding vector or error dictionary
    """
    if model_type.lower() != "vit":
        return {"error": f"Unsupported model type: {model_type}. Currently only 'vit' is supported."}
    
    # Check cache first
    cache_key = f"{image_path}_{model_type}"
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    try:
        # Open and convert image to RGB
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Process image with ViT
            inputs = vit_processor(images=img, return_tensors="pt")
            
            with torch.no_grad():
                outputs = vit_model(**inputs)
                
                # Get both pooled output and last hidden states
                pooled_output = outputs.pooler_output
                last_hidden_states = outputs.last_hidden_state
                
                # Combine pooled output with average of last hidden states for richer representation
                avg_hidden_states = torch.mean(last_hidden_states, dim=1)
                combined_features = torch.cat([pooled_output, avg_hidden_states], dim=1)
                
                # Normalize the combined features
                normalized_features = F.normalize(combined_features, p=2, dim=1)
                
                # Cache the result
                embedding = normalized_features.squeeze().tolist()
                EMBEDDING_CACHE[cache_key] = embedding
                
                # Save cache periodically
                if len(EMBEDDING_CACHE) % SAVE_FREQUENCY == 0:
                    save_cache()
                    
                return embedding
    
    except Exception as e:
        error_dict = {"error": f"Failed to compute embedding: {str(e)}"}
        EMBEDDING_CACHE[cache_key] = error_dict
        return error_dict

# Load cache at module import
load_cache()

def get_hierarchical_similarity(embedding1: List[float], embedding2: List[float], 
                              level_weights: Dict[str, float] = None) -> float:
    """
    Calculate similarity between embeddings with optional weighting for hierarchical levels.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        level_weights: Optional weights for different hierarchical levels
    
    Returns:
        Similarity score between 0 and 1
    """
    if level_weights is None:
        # Default weights prioritizing higher taxonomic levels
        level_weights = {
            'kingdom': 1.0,
            'phylum': 0.9,
            'class': 0.8,
            'order': 0.7,
            'family': 0.6,
            'genus': 0.5,
            'species': 0.4
        }
    
    # Convert to tensors
    e1 = torch.tensor(embedding1)
    e2 = torch.tensor(embedding2)
    
    # Calculate cosine similarity
    similarity = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
    
    return float(similarity)

# Example usage
if __name__ == "__main__":
    image_path = "Overview.png"
    
    # Get embedding using ViT
    vit_embedding = get_image_embedding(image_path, model_type="vit")
    
    if isinstance(vit_embedding, dict) and "error" in vit_embedding:
        print(f"ViT Error: {vit_embedding['error']}")
    else:
        print(f"ViT Embedding shape: {len(vit_embedding)}")
        print(f"ViT First few values: {vit_embedding[:5]}")
    
    # Save cache before exit
    save_cache()