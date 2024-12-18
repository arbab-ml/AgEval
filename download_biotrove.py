import os
import requests
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import time

def is_valid_url(url):
    """Check if a URL is valid."""
    return url is not None and isinstance(url, str) and url.startswith(('http://', 'https://'))

def download_biotrove(num_records=100, batch_size=50):
    """
    Downloads the BioTrove dataset from Hugging Face and organizes it into a local directory structure.
    
    Args:
        num_records (int): Number of records to download. Defaults to 100.
        batch_size (int): Number of records to process in each batch. Defaults to 50.
    """
    # Create base directory and images subdirectory
    base_dir = "biotrove-data"
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print("Loading BioTrove dataset from Hugging Face...")
    try:
        dataset = load_dataset("BGLab/BioTrove", streaming=True)
        train_data = dataset['train']
        
        all_data = []
        processed_count = 0
        skipped_count = 0
        
        for item in tqdm(train_data.take(num_records + 100), total=num_records, desc="Processing dataset"):
            try:
                if processed_count >= num_records:
                    break
                    
                # Create data entry with all original fields
                data_entry = dict(item)
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
            
            # Save metadata periodically
            if len(all_data) % batch_size == 0:
                metadata_df = pd.DataFrame(all_data)
                metadata_path = os.path.join(base_dir, "metadata.csv")
                metadata_df.to_csv(metadata_path, index=False)
        
        # Final metadata save
        if all_data:
            metadata_df = pd.DataFrame(all_data)
            metadata_path = os.path.join(base_dir, "metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            
            print("\nDownload complete!")
            print(f"Images saved to: {os.path.abspath(images_dir)}")
            print(f"Metadata saved to: {os.path.abspath(metadata_path)}")
            print(f"\nSkipped {skipped_count} items due to invalid URLs")
            
            # Print dataset statistics
            print("\nDataset statistics:")
            print(f"Total entries: {len(metadata_df)}")
            print(f"Successfully downloaded images: {len(metadata_df[metadata_df['local_url'].notna()])}")
            print("\nClass distribution:")
            print(metadata_df['common_name'].value_counts())
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting BioTrove dataset download...")
    success = download_biotrove(num_records=200)
    if success:
        print("Dataset download and organization completed successfully.")
    else:
        print("Dataset download failed. Please check the error messages above.")