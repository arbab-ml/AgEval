import requests
import base64
import os
from typing import List, Union
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, ViTModel

# CLIP model setup
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

# ViT model setup
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
vit_model = ViTModel.from_pretrained(VIT_MODEL_NAME)

def get_image_embedding(image_path: str, model_type: str = "clip") -> Union[List[float], dict]:
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB mode if it's not
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            if model_type.lower() == "clip":
                # Process the image using CLIP processor
                inputs = clip_processor(images=img, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                
                # Convert to list and return
                return image_features.squeeze().tolist()
            
            elif model_type.lower() == "vit":
                # Process the image using ViT processor
                inputs = vit_processor(images=img, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    outputs = vit_model(**inputs)
                
                # Use pooler_output as the image embedding
                image_embedding = outputs.pooler_output
                
                # Convert to list and return
                return image_embedding.squeeze().tolist()
            
            else:
                return {"error": f"Unsupported model type: {model_type}"}
    
    except IOError:
        return {"error": f"Unable to open or process the image at {image_path}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Usage
image_path = "/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib/Overview.png"

# Get embedding using CLIP
clip_embedding = get_image_embedding(image_path, model_type="clip")

if isinstance(clip_embedding, dict) and "error" in clip_embedding:
    print(f"CLIP Error: {clip_embedding['error']}")
else:
    print(f"CLIP Embedding shape: {len(clip_embedding)}")
    print(f"CLIP First few values: {clip_embedding[:5]}")

# Get embedding using ViT
vit_embedding = get_image_embedding(image_path, model_type="vit")

if isinstance(vit_embedding, dict) and "error" in vit_embedding:
    print(f"ViT Error: {vit_embedding['error']}")
else:
    print(f"ViT Embedding shape: {len(vit_embedding)}")
    print(f"ViT First few values: {vit_embedding[:5]}")

# Note: This implementation uses the CLIP and ViT models directly from Hugging Face Transformers,
# which is more efficient and doesn't require making API calls.
# Make sure to install the required packages:
# pip install transformers torch Pillow