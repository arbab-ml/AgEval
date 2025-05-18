import os
import requests
import pandas as pd
import time
from tqdm import tqdm

def is_valid_url(url):
    """Check if a URL is valid."""
    return url is not None and isinstance(url, str) and url.startswith(('http://', 'https://'))

def process_dataset(dataset_path):
    """
    Expects `dataset_path` to be a path to a parquet file.
    """
    
    # Create directories
    base_dir = os.path.join("processed_data", dataset_path.split(".")[0].split("/")[-1])
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    try:        
        print(f"Loading {dataset_path} dataset...")
        df = pd.read_parquet(dataset_path)
        print(f"Loaded {len(df)} samples")
        print("\nClass distribution:")
        print(df['common_name'].value_counts())
        
        # Download images
        processed_count = 0
        skipped_count = 0
        all_data = []
        
        for _, item in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
            try:
                # Create data entry with all original fields
                data_entry = item.to_dict()
                image_filename = f"{item['photo_id']}.jpg"
                data_entry['local_url'] = os.path.join('images', image_filename) if is_valid_url(item['photo_url']) else None
                all_data.append(data_entry)
                
                # Skip download for invalid URLs
                if not is_valid_url(item['photo_url']):
                    skipped_count += 1
                    continue
                
                # Define image path
                image_path = os.path.join(images_dir, image_filename)
                
                # Skip if file already exists
                if os.path.exists(image_path):
                    processed_count += 1
                    continue
                
                # Download image with retry mechanism
                max_retries = 3
                retry_count = 0
                download_success = False
                
                while retry_count < max_retries and not download_success:
                    try:
                        response = requests.get(item['photo_url'], timeout=30)
                        if response.status_code == 200:
                            with open(image_path, 'wb') as f:
                                f.write(response.content)
                            download_success = True
                        else:
                            print(f"\nAttempt {retry_count + 1}: Failed to download image {item['photo_id']}: HTTP {response.status_code}")
                            retry_count += 1
                            time.sleep(1)
                    except requests.exceptions.RequestException as e:
                        print(f"\nAttempt {retry_count + 1}: Network error for image {item['photo_id']}: {str(e)}")
                        retry_count += 1
                        time.sleep(1)
                
                if download_success:
                    processed_count += 1
                    # Add small delay between successful downloads
                    time.sleep(0.1)
                else:
                    # If download failed after all retries, set local_url to None
                    all_data[-1]['local_url'] = None
                
            except Exception as e:
                print(f"\nError processing item: {str(e)}")
                continue
        
        # Save metadata
        metadata_df = pd.DataFrame(all_data)
        metadata_path = os.path.join(base_dir, "balanced_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        
        print("\nDownload complete!")
        print(f"Images saved to: {os.path.abspath(images_dir)}")
        print(f"Metadata saved to: {os.path.abspath(metadata_path)}")
        print(f"\nSkipped {skipped_count} items due to invalid URLs")
        
        # Print dataset statistics
        print("\nDataset statistics:")
        print(f"Total entries: {len(metadata_df)}")
        print(f"Successfully downloaded images: {len(metadata_df[metadata_df['local_url'].notna()])}")
        print("\nFinal class distribution:")
        print(metadata_df['common_name'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

def main():
    data_path = "STAGE_BioTrove_data/plantae_group_filtered.parquet"
    success = process_dataset(data_path)
    
    if success:
        print("Dataset download and organization completed successfully.")
    else:
        print("Dataset download failed. Please check the error messages above.")

if __name__ == "__main__":
    main()