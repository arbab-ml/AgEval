import os
from huggingface_hub import hf_hub_download


def download_balanced_dataset(repo_id, filename):
    
    local_dir = os.path.join("STAGE_BioTrove_data", filename.split(".")[0].split("/")[-1])

    parquet_path = hf_hub_download(repo_id=repo_id, 
                                     filename=filename,
                                     repo_type="dataset",
                                     local_dir=local_dir)
    
    return parquet_path

def main():
    print("Starting BioTrove-Balanced dataset download...")
    success = download_balanced_dataset(repo_id="BGLab/BioTrove-Train", filename="BioTrove-benchmark/BioTrove-Balanced.parquet")
    
    if success:
        print("Dataset download and organization completed successfully.")
    else:
        print("Dataset download failed. Please check the error messages above.")

if __name__ == "__main__":
    main()