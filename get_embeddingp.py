import requests
import base64
import os
from typing import List, Union
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, ViTModel, ResNetModel, ResNetConfig, AutoFeatureExtractor

AVAILABLE_ENCODERS = ["vit", "clip", "resnet"]
# CLIP model setup
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

# ViT model setup
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
vit_model = ViTModel.from_pretrained(VIT_MODEL_NAME)

# ResNet model setup
RESNET_MODEL_NAME = "microsoft/resnet-50"
resnet_feature_extractor = AutoFeatureExtractor.from_pretrained(RESNET_MODEL_NAME)
resnet_model = ResNetModel.from_pretrained(RESNET_MODEL_NAME)

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
                # Process th