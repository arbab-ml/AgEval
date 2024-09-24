

# total_samples_to_check = 10
# vendor, model, model_name =all_vendors_models[0].values() # index 2 is gemini, 3 is llava

# load all the modules local again instead of cache
# %load_ext autoreload
# %autoreload 2
# Import the required libraries
import os
import json
import base64
import asyncio
import aiohttp
import time
from anthropic import Anthropic
from PIL import Image
import io
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
import nest_asyncio
from tqdm import tqdm
import re
from data_loader import load_and_prepare_data_SBRD, load_and_prepare_data_DurumWheat, load_and_prepare_data_soybean_seeds, load_and_prepare_data_mango_leaf, load_and_prepare_data_DeepWeeds, load_and_prepare_data_IP02, load_and_prepare_data_bean_leaf, load_and_prepare_data_YellowRust, load_and_prepare_data_FUSARIUM22, load_and_prepare_data_InsectCount, load_and_prepare_data_DiseaseQuantify, load_and_prepare_data_IDC, load_and_prepare_data_Soybean_PNAS, load_and_prepare_data_Soybean_Dangerous_Insects
nest_asyncio.apply()
global vision_prompt

#claude-3-sonnet-20240229
all_vendors_models=[
    {"vendor": "openai", "model": "gpt-4o-2024-05-13", "model_name": "GPT-4o"}, #  done
    # {"vendor": "anthropic", "model": "claude-3-5-sonnet-20240620", "model_name": "Claude-3.5-sonnet"}, #done 
    # {"vendor": "anthropic", "model": "claude-3-haiku-20240307", "model_name": "Claude-3-haiku"}, #done 
    # {"vendor": "openrouter", "model": "liuhaotian/llava-yi-34b", "model_name": "LLaVA v1.6 34B"}, #done
    # {"vendor": "google", "model": "gemini-1.5-flash-latest", "model_name": "Gemini-flash-1.5"}, #done
    # {"vendor": "google", "model": "gemini-1.5-pro", "model_name": "Gemini-pro-1.5"},#done 
]

universal_prompt = """
Given the image, identify the class. Use the following list of possible classes for your prediction It should be one of the : {expected_classes}. Be attentive to subtle details as some classes may appear similar. Provide your answer in the following JSON format:
{{"prediction": "class_name"}}
Replace "class_name" with the appropriate class from the list above based on your analysis of the image.
The labels should be entered exactly as they are in the list above i.e., {expected_classes}.
The response should start with {{ and contain only a JSON object (as specified above) and no other text.
"""
insect_count_prompt="""
    Analyze this image of a yellow sticky insect trap. Count the total number of visible insects caught on the trap. Only look for insects which are easily visible to nacked eye and look bigger compared to the other background artifacts.  Provide your answer in the following JSON format:
    {{"prediction": "number"}}
    Replace "number" with your best estimate of the total insect count based on your analysis of the image.
    The number should be entered exactly as a whole number (without any symbols) in a range of {expected_classes}
    The response should start with {{ and contain only a JSON object (as specified above) and no other text.
    """
disease_count_prompt="""
    Analyze this image of a leaf to get the total percentage of affected leaf. The images are of several plant leaf-like Apple Scab Leaf, Apple rust leaf, Bell_pepper leaf spot, Corn leaf blight, Potato leaf early blight, etc. The affected area is: diseased leaf area / total image area. Provide your answer in the following JSON format:
    {{"prediction": "number"}}
    Replace "number" with your best estimate of the percent on your analysis of the image.
    The number should be entered exactly as a whole number (without any symbols) in a range of {expected_classes}
    The response should start with {{ and contain only a JSON object (as specified above) and no other text.
    """
idc_prompt="""
    Analyze this image of a soybean canopy to determine the iron deficiency chlorosis (IDC) severity rating. The images are of soybean plants exhibiting various levels of IDC symptoms, ranging from healthy green plants to those with severe chlorosis and necrosis. Evaluate the extent of yellowing and browning in the canopy. Provide your answer in the following JSON format:
    {{"prediction": "number"}}
    Replace "number" with your best estimate of the IDC severity rating based on your analysis of the image.
    The number should be entered exactly as a whole number (without any symbols) in a range of {expected_classes}. Higher value means more severity.
    The response should start with {{ and contain only a JSON object (as specified above) and no other text.
    """


universal_shots= [8, 4, 2, 1, 0]
# only 1 and 0 shots

datasets = [
    # {"loader": load_and_prepare_data_SBRD, "samples": 100, "shots": universal_shots, "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_DurumWheat, "samples": 100, "shots": universal_shots, "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_soybean_seeds, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_mango_leaf, "samples": 50, "shots": universal_shots,  "vision_prompt": universal_prompt},
    {"loader": load_and_prepare_data_DeepWeeds, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt},
    # # {"loader": load_and_prepare_data_IP02, "samples": 105, "shots": universal_shots,  "vision_prompt": universal_prompt}, # implement resizing for this data and run every model again
    # {"loader": load_and_prepare_data_bean_leaf, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_YellowRust, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_FUSARIUM22, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt},
    # {"loader": load_and_prepare_data_InsectCount, "samples": 100, "shots": universal_shots,  "vision_prompt": insect_count_prompt}, 
    # {"loader": load_and_prepare_data_DiseaseQuantify, "samples": 100, "shots": universal_shots,  "vision_prompt": disease_count_prompt},
    # {"loader": load_and_prepare_data_IDC, "samples": 100, "shots": universal_shots,  "vision_prompt": idc_prompt},
    # {"loader": load_and_prepare_data_Soybean_PNAS, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt}, 
    # {"loader": load_and_prepare_data_Soybean_Dangerous_Insects, "samples": 100, "shots": universal_shots,  "vision_prompt": universal_prompt}, 

]



vision_prompt = ""
def extract_json(s):
    """Extract the first JSON object from a string."""
    json_match = re.search(r'\{.*\}', s, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None

def load_image(image_path: str) -> str:
    """
    Load image from file, convert to JPEG, and encode as base64.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = []

    async def wait(self):
        while True:
            current_time = time.time()
            self.request_times = [t for t in self.request_times if t > current_time - self.time_window]
            if len(self.request_times) < self.max_requests:
                self.request_times.append(current_time)
                break
            await asyncio.sleep(0.1)

class GPTAPI:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.rate_limiter = RateLimiter(max_requests=20, time_window=1)

    async def get_image_information(self, inputs: dict) -> str:
        await self.rate_limiter.wait()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": inputs['prompt']},
                        *inputs['examples'],
                        {"type": "text", "text": inputs['prompt']},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096, 
            "temperature":1.0
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=payload) as response:
                result = await response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]['message']['content']
                else:
                    raise Exception(f"Unexpected API response format: {result}")

class ClaudeAPI:
    def __init__(self, api_key, model):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(max_requests=5, time_window=2)  # Adjust these values as needed

    async def get_image_information(self, inputs: dict) -> str:
        await self.rate_limiter.wait()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": inputs['prompt']},
                    *[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": ex['image_url']['url'].split(',')[1] if ex['type'] == 'image_url' else ex['source']['data']
                            }
                        }
                        if ex['type'] in ['image_url', 'image'] else ex
                        for ex in inputs['examples']
                    ],
                    {"type": "text", "text": inputs['prompt']},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": inputs['image']
                        }
                    }
                ]
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=1.0,
            messages=messages
        )
        return response.content[0].text
# Set the base directory


class OpenRouterAPI:
    def __init__(self, api_key, model):#  liuhaotian/llava-yi-34b
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.rate_limiter = RateLimiter(max_requests=15, time_window=5)  # Adjust as needed

    async def get_image_information(self, inputs: dict) -> str:
        await self.rate_limiter.wait()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": inputs['prompt']},
                        *inputs['examples'],
                        {"type": "text", "text": inputs['prompt']}, 
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}"
                            }
                        }
                    ]
                }
            ],
            "temperature":1.0
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, json=payload) as response:
                result = await response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]['message']['content']
                else:
                    raise Exception(f"Unexpected API response format: {result}")

class GeminiAPI:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.headers = {
            "Content-Type": "application/json",
        }
        self.rate_limiter = RateLimiter(max_requests=15, time_window=5)  # Adjust as needed

    async def get_image_information(self, inputs: dict) -> str:
        await self.rate_limiter.wait()
        
        gemini_examples = []
        gemini_examples.extend([{"text": inputs['prompt']}])
        for example in inputs['examples']:
            if 'image_url' in example:
                gemini_examples.extend([
                    {"inline_data": {"mime_type": "image/jpeg", "data": example['image_url']['url'].split(',')[1]}}
                ])
            elif 'text' in example:
                gemini_examples.append({"text": example['text']})

        # Add the final prompt and image
        gemini_examples.extend([
            {"inline_data": {"mime_type": "image/jpeg", "data": inputs['image']}}
        ])

        payload = {
            "contents": [
                {
                    "parts": gemini_examples
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 4096,
                "response_mime_type": "application/json",
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}?key={self.api_key}", headers=self.headers, json=payload) as response:
                result = await response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]['content']['parts'][0]['text']
                else:
                    raise Exception(f"Unexpected API response format: {result}")

class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Processing images")

    def update(self):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()

################################################################################################################################################################

# Add this import at the top of the file
from get_embeddingp import get_image_embedding, AVAILABLE_ENCODERS

# Remove the previously added CLIP-related imports and functions

# Modify the precompute_embeddings function
def precompute_embeddings(all_data, encoder):
    embeddings = {}
    print(f"Precomputing {encoder} embeddings...")
    for idx, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Computing embeddings"):
        image_path = row[0]
        embedding = get_image_embedding(image_path, model_type=encoder)
        if isinstance(embedding, dict) and "error" in embedding:
            print(f"Error computing embedding for {image_path}: {embedding['error']}")
        else:
            embeddings[idx] = embedding
    return embeddings

# Modify the process_image function
async def process_image(api, i, number_of_shots, all_data_results, all_data, progress_bar, embeddings, use_embedding=True, encoder="vit"):
    try:
        image_path = all_data[0][i]
        image_base64 = load_image(image_path)
        if image_base64 is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        examples = []
        example_paths = []
        example_categories = []
        
        if use_embedding:
            # Use adaptive example selection
            input_embedding = get_image_embedding(image_path, model_type=encoder)
            if isinstance(input_embedding, dict) and "error" in input_embedding:
                raise ValueError(f"Failed to compute embedding for {image_path}: {input_embedding['error']}")
            
            similarities = []
            for idx, embedding in embeddings.items():
                if idx != i:  # Exclude the current image
                    similarity = np.dot(input_embedding, embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(embedding))
                    similarities.append((idx, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_indices = [idx for idx, _ in similarities[:number_of_shots]]
        else:
            # Random selection
            similar_indices = random.sample([idx for idx in range(len(all_data)) if idx != i], number_of_shots)

        # Count examples from the same category
        same_category_count = 0
        input_category = all_data.at[i, 1]

        for j in similar_indices:
            example_image_path = all_data[0][j]
            example_category = all_data.at[j, 1]
            example_image_base64 = load_image(example_image_path)
            
            if example_image_base64 is None:
                print(f"Failed to load example image: {example_image_path}")
                continue

            if example_category == input_category:
                same_category_count += 1

            if isinstance(api, GPTAPI) or isinstance(api, OpenRouterAPI):
                examples.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image_base64}", "detail": "high"}})
            elif isinstance(api, ClaudeAPI):
                examples.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": example_image_base64
                    }
                })
            elif isinstance(api, GeminiAPI):
                examples.append({"image_url": {"url": f"data:image/jpeg;base64,{example_image_base64}"}})
            else:
                raise ValueError(f"Unsupported API type: {type(api)}")

            examples.append({"type": "text", "text": f'{{"prediction": "{example_category}"}}' })
            example_paths.append(example_image_path)
            example_categories.append(example_category)

        prediction = await api.get_image_information({
            "image": image_base64, 
            "examples": examples, 
            "prompt": vision_prompt
        })
        
        try:
            extracted_json = extract_json(prediction)
            parsed_prediction = extracted_json['prediction']
        except Exception as e:
            print(f"Error parsing JSON for image {image_path}. API response: {prediction}. Error: {str(e)}")
            parsed_prediction = 'NA'

        prefix = "Embedding" if use_embedding else "Random"
        all_data_results.at[i, f"{prefix} # of Shots {number_of_shots}"] = parsed_prediction
        all_data_results.at[i, f"{prefix} Example Paths {number_of_shots}"] = str(example_paths)
        all_data_results.at[i, f"{prefix} Example Categories {number_of_shots}"] = str(example_categories)
        all_data_results.at[i, f"{prefix} Same Category Count {number_of_shots}"] = same_category_count

    except Exception as e:
        print(f"Error processing {all_data[0][i]}: {str(e)}")
        prefix = "Embedding" if use_embedding else "Random"
        all_data_results.at[i, f"{prefix} # of Shots {number_of_shots}"] = 'NA'
        all_data_results.at[i, f"{prefix} Example Paths {number_of_shots}"] = 'NA'
        all_data_results.at[i, f"{prefix} Example Categories {number_of_shots}"] = 'NA'
        all_data_results.at[i, f"{prefix} Same Category Count {number_of_shots}"] = 'NA'
    finally:
        progress_bar.update()

async def process_images_for_shots(api, number_of_shots, all_data_results, all_data, embeddings, encoder):
    progress_bar = ProgressBar(len(all_data) * 2)
    tasks = []
    for i in range(len(all_data)):
        task_embedding = asyncio.ensure_future(process_image(api, i, number_of_shots, all_data_results, all_data, progress_bar, embeddings, use_embedding=True, encoder=encoder))
        task_random = asyncio.ensure_future(process_image(api, i, number_of_shots, all_data_results, all_data, progress_bar, embeddings, use_embedding=False, encoder=encoder))
        tasks.extend([task_embedding, task_random])
    
    await asyncio.gather(*tasks)
    progress_bar.close()

# Update the main function
async def main():
    global vision_prompt

    for dataset in datasets:
        loader = dataset["loader"]
        total_samples_to_check = dataset["samples"]
        shots = dataset["shots"]
        
        all_data, expected_classes, output_file_name = loader(total_samples_to_check)
        print(f"Dataset Name: {output_file_name}")
        print(f"Number of classes / unique labels: {len(expected_classes)}")
        print(f"Expected classes: {expected_classes}")
        print(f"Class distribution: {all_data[1].value_counts()}")
        print("----------------------------")
        vision_prompt = dataset["vision_prompt"].format(expected_classes=expected_classes)

        for encoder in AVAILABLE_ENCODERS:
            print(f"\nProcessing dataset: {output_file_name} with encoder: {encoder}")
            embeddings = precompute_embeddings(all_data, encoder)
            
            for vendor_model in all_vendors_models:
                vendor = vendor_model["vendor"]
                model = vendor_model["model"]
                model_name = vendor_model["model_name"]

                print(f"Running model: {model_name}")

                if vendor == "openai":
                    api = GPTAPI(api_key=os.getenv("OPENAI_API_KEY"), model=model)
                elif vendor == "anthropic":
                    api = ClaudeAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model)
                elif vendor == "openrouter":
                    api = OpenRouterAPI(api_key=os.getenv("OPENROUTER_API_KEY"), model=model)
                elif vendor == "google":
                    api = GeminiAPI(api_key=os.getenv("GOOGLE_API_KEY"), model=model)
                else:
                    raise ValueError(f"Unsupported model type: {vendor}")

                all_data_results = all_data.copy(deep=True)
                all_data_results.columns = all_data_results.columns.map(str)

                for number_of_shots in shots:
                    print(f"Running with {number_of_shots} shots")
                    await process_images_for_shots(api, number_of_shots, all_data_results, all_data, embeddings, encoder)

                    embedding_accuracy = calculate_accuracy(all_data_results, f"Embedding # of Shots {number_of_shots}")
                    random_accuracy = calculate_accuracy(all_data_results, f"Random # of Shots {number_of_shots}")
                    print(f"Accuracy for {number_of_shots}-shot (Embedding): {embedding_accuracy:.2f}")
                    print(f"Accuracy for {number_of_shots}-shot (Random): {random_accuracy:.2f}")

                    # Detailed logging
                    print("\nDetailed results:")
                    for i in range(len(all_data)):
                        true_label = all_data.at[i, 1]
                        embedding_prediction = all_data_results.at[i, f"Embedding # of Shots {number_of_shots}"]
                        random_prediction = all_data_results.at[i, f"Random # of Shots {number_of_shots}"]
                        print(f"Image {i}: True: {true_label}, Embedding Prediction: {embedding_prediction}, Random Prediction: {random_prediction}")

                results_dir = os.path.join("results", model_name, encoder)
                os.makedirs(results_dir, exist_ok=True)
                output_file = os.path.join(results_dir, f"{output_file_name}.csv")
                all_data_results.to_csv(output_file)
                print(f"Results saved to {output_file}")

        print(f"Completed processing for dataset: {output_file_name}\n")

# Update the calculate_accuracy function
def calculate_accuracy(all_data_results, column_name):
    correct = sum(all_data_results['1'] == all_data_results[column_name])
    total = len(all_data_results)
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    asyncio.run(main())