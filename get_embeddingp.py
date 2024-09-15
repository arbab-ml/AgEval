import requests
import base64
import os
from typing import List, Union
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

def get_image_embedding(image_path: str) -> Union[List[float], dict]:
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB mode if it's not
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Process the image using CLIP processor
            inputs = processor(images=img, return_tensors="pt")
            
            # Get image features
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Convert to list and return
            return image_features.squeeze().tolist()
    except IOError:
        return {"error": f"Unable to open or process the image at {image_path}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Usage
image_path = "/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib/Overview.png"
embedding = get_image_embedding(image_path)

if isinstance(embedding, dict) and "error" in embedding:
    print(f"Error: {embedding['error']}")
else:
    print(f"Embedding shape: {len(embedding)}")
    print(f"First few values: {embedding[:5]}")

# Note: This implementation uses the CLIP model directly from Hugging Face Transformers,
# which is more efficient and doesn't require making API calls.
# Make sure to install the required packages:
# pip install transformers torch Pillow