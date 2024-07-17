import os
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
import numpy as np

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil
from tqdm import tqdm
import difflib
import requests
from tqdm import tqdm


# Utility functions
def download_with_progress(dataset_name, path="."):
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset_name}")
    api.dataset_download_files(dataset_name, path=path, unzip=True, quiet=False)
    print("Download complete.")

def get_closest_match(name, options):
    return difflib.get_close_matches(name, options, n=1, cutoff=0.2)[0]

def rename_folders(base_directory, expected_classes):
    print("Renaming folders to match expected classes...")
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            try:
                closest_match = get_closest_match(folder, expected_classes)
                new_folder_path = os.path.join(base_directory, closest_match)
                os.rename(folder_path, new_folder_path)
                print(f"Renamed '{folder}' to '{closest_match}'")
            except IndexError:
                print(f"Warning: No close match found for '{folder}'. Skipping rename.")

def rename_folders_dict(base_directory, rename_dict):
    print("Renaming folders based on provided dictionary...")
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            if folder in rename_dict:
                new_folder_name = rename_dict[folder]
                new_folder_path = os.path.join(base_directory, new_folder_name)
                os.rename(folder_path, new_folder_path)
                print(f"Renamed '{folder}' to '{new_folder_name}'")
            else:
                print(f"Warning: No mapping found for '{folder}'. Skipping rename.")

def convert_tiff_to_jpg(file_path):
    if file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif'):
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert('RGB')
                jpg_path = os.path.splitext(file_path)[0] + '.jpg'
                rgb_img.save(jpg_path, 'JPEG')
            os.remove(file_path)  # Remove the original TIFF file
            return jpg_path
        except Exception as e:
            print(f"Error converting {file_path}: {str(e)}")
            return file_path
    return file_path
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def get_file_urls(record_id):
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        data = response.json()
        return [(file['links']['self'], file['key']) for file in data.get('files', [])]
    else:
        print(f"Error fetching metadata. Status code: {response.status_code}")
        return []

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the total number of files in the zip
        total_files = len(zip_ref.infolist())
        
        # Extract all files with a progress bar
        for file in tqdm(iterable=zip_ref.infolist(), total=total_files, desc="Extracting"):
            zip_ref.extract(member=file, path=extract_path)

# Load and prepare data functions
def load_and_prepare_data_SBRD(total_samples_to_check):
    # Dataset details
    dataset_name = "isaacritharson/severity-based-rice-leaf-diseases-dataset"
    download_path = "./data/Severity_Based_Rice_Leaf_Diseases_Dataset"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")

    # The dataset is downloaded and extracted to the specified directory
    base_directory =os.path.join(download_path, "Leaf Disease Dataset", "train")

    expected_classes = ['Healthy', 'Mild Bacterial Blight', 'Mild Blast', 'Mild Brownspot', 'Mild Tungro', 'Severe Bacterial Blight', 'Severe Blast', 'Severe Brownspot', 'Severe Tungro']
    
    # Rename folders to match expected classes
    rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        #if subdir == ".DS_Store":
        #    continue
        subdir_path = os.path.join(base_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "SBRD"


def load_and_prepare_data_DurumWheat(total_samples_to_check):
      # Dataset details
    dataset_name = "muratkokludataset/durum-wheat-dataset"
    download_path = "./data/Durum_Wheat_Dataset"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")

    # The dataset is downloaded and extracted to the specified directory
    base_directory =os.path.join(download_path, "Durum_Wheat_Dataset", "Dataset2-Durum Wheat Video Images")

    expected_classes = ['Foreign Matters', 'Starchy Kernels', 'Vitreous Kernels']
    # Rename folders to match expected classes
      # Define the renaming dictionary
    rename_dict = {
        '1-Images from Vitreous Durum Wheat': 'Vitreous Kernels',
        '2-Images from Starchy Durum Wheat': 'Starchy Kernels',
        '3-Images from Foreign Matters': 'Foreign Matters'
    }

    # Rename folders using the dictionary
    rename_folders_dict(base_directory, rename_dict)

    #rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []
    
    #print("Processing data and converting TIFF to JPG...")
    for subdir in tqdm(os.listdir(base_directory), desc="Processing classes"):
        subdir_path = os.path.join(base_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                file_path = os.path.join(subdir_path, filename)
                converted_path = convert_tiff_to_jpg(file_path)
                file_paths.append(converted_path)
                labels.append(subdir)
    
    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Durum Wheat"


def load_and_prepare_data_soybean_seeds(total_samples_to_check):


     # Dataset details
    dataset_name = "warcoder/soyabean-seeds"
    download_path = "./data/soyabean-seeds_Dataset"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'Soybean Seeds')

    #base_directory = 'data/Soybean Seeds'
    expected_classes = ['Broken', 'Immature', 'Intact', 'Skin-damaged', 'Spotted']
    # Rename folders to match expected classes
    rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Soybean Seeds"

def load_and_prepare_data_mango_leaf(total_samples_to_check):
      
    # Dataset details
    dataset_name = "aryashah2k/mango-leaf-disease-dataset"
    download_path = "./data/mango-leaf-disease-dataset"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path)


    expected_classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    
    rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []
    
    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)
    
    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Mango Leaf Disease"

def load_and_prepare_data_DeepWeeds(total_samples_to_check):

    # Dataset details
    dataset_name = "imsparsh/deepweeds"
    download_path = "./data/deepweeds"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path)

    expected_classes = ['Chinee apple', 'Lantana', 'Negative', 'Snake weed', 'Siam weed', 'Prickly acacia', 'Parthenium', 'Rubber vine', 'Parkinsonia']
    #rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    
    # Load the labels CSV file
    labels_df = pd.read_csv(os.path.join(base_directory, 'labels', 'labels.csv'))
    
    # Create a dataframe with file paths and labels
    data = pd.DataFrame({
        0: labels_df['Filename'].apply(lambda x: os.path.join(base_directory, 'images', x)),
        1: labels_df['Species']
    })
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "DeepWeeds"


def load_and_prepare_data_IP02(total_samples_to_check):
    
    dataset_name = "rtlmhjbn/ip02-dataset"
    download_path = "./data/ip02-dataset"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'classification','train')
    classes_file = os.path.join(download_path, 'classes.txt')



    def read_classes(file_path):
        classes = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    class_id, class_name = parts
                    classes[int(class_id) - 1] = class_name.strip()  # Subtract 1 to match folder names
        return classes

    expected_classes = read_classes(classes_file)
    samples_per_class = int(total_samples_to_check / len(expected_classes))

    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_paths.append(os.path.join(subdir_path, filename))
                    labels.append(expected_classes[int(subdir)])

    data = pd.DataFrame({0: file_paths, 1: labels})

    # Use a fixed random state for deterministic sampling
    random_state = 42

    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes.values():
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)

    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), list(expected_classes.values()), "IP02"



def load_and_prepare_data_bean_leaf(total_samples_to_check):
    # Dataset details
    dataset_name = "marquis03/bean-leaf-lesions-classification"
    download_path = "./data/bean-leaf-lesions-classification"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'train')
    expected_classes = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']
    rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    
    file_paths = []
    labels = []
    
    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)
    
    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Bean Leaf Lesions"


def load_and_prepare_data_YellowRust(total_samples_to_check):
    dataset_name = "tolgahayit/yellowrust19-yellow-rust-disease-in-wheat"
    download_path = "./data/yellowrust19-yellow-rust-disease-in-wheat"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'YELLOW-RUST-19','YELLOW-RUST-19')
   
    expected_classes = ['Moderately Resistant (MR)', 'Moderately Susceptible (MS)', 'MRMS', 'No disease (0)', 'Resistant (R)', 'Susceptible (S)']
    
    rename_dict = {
        'MR':'Moderately Resistant (MR)', 
        'MS':'Moderately Susceptible (MS)', 
        'MRMS':'MRMS',
        '0': 'No disease (0)',
        'R': 'Resistant (R)',
        'S': 'Susceptible (S)'
    }

    rename_folders_dict(base_directory, rename_dict )

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)

        if os.path.isdir(subdir_path):  # Check if it's a directory
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_paths.append(os.path.join(subdir_path, filename))
                    labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Yellow Rust 19"


def load_and_prepare_data_FUSARIUM22(total_samples_to_check):
    dataset_name = "tolgahayit/fusarium-wilt-disease-in-chickpea-dataset"
    download_path = "./data/fusarium-wilt-disease-in-chickpea-dataset"

    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'FUSARIUM-22','dataset_raw')
   
    expected_classes = ['Highly Resistant', 'Highly Susceptible', 'Moderately Resistant', 'Resistant', 'Susceptible']
    
    rename_dict = {
        '1(HR)':'Highly Resistant', 
        '9(HS)':'Highly Susceptible', 
        '5(MR)':'Moderately Resistant',
        '3(R)': 'Resistant', 
        '7(S)':'Susceptible'        
    }

    rename_folders_dict(base_directory, rename_dict )

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "FUSARIUM 22"


def load_and_prepare_data_DiseaseQuantify(total_samples_to_check):

    dataset_name = "sovitrath/leaf-disease-segmentation-with-trainvalid-split"
    download_path = "./data/leaf-disease-segmentation-with-trainvalid-split"

    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'leaf_disease_segmentation' ,'orig_data')
    
    images_dir = os.path.join(base_directory, 'train_images')
    masks_dir = os.path.join(base_directory, 'train_masks')
    
    file_paths = []
    labels = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                # Open the mask image
                with Image.open(mask_path) as mask:
                    # Convert to numpy array
                    mask_array = np.array(mask)
                    
                    
                    # Check if the image is RGB
                    if len(mask_array.shape) == 3:
                        # Identify red pixels (R > 0, G = 0, B = 0)
                        affected_pixels = (mask_array[:,:,0] > 0) & (mask_array[:,:,1] == 0) & (mask_array[:,:,2] == 0)
                    else:
                        # For grayscale images, consider any non-zero pixel as affected
                        affected_pixels = mask_array > 0
                    
                    # Calculate percentage of affected area
                    total_pixels = mask_array.shape[0] * mask_array.shape[1]
                    percentage_affected = int((np.sum(affected_pixels) / total_pixels) * 100)
                    
                file_paths.append(image_path)
                labels.append(percentage_affected)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    # Sample the data
    if len(data) > total_samples_to_check:
        sampled_data = data.sample(n=total_samples_to_check, random_state=random_state)
    else:
        sampled_data = data
        print(f"Warning: Not enough samples. Using all {len(data)} available samples.")
    
    # Shuffle the sampled data
    shuffled_data = shuffle(sampled_data, random_state=random_state).reset_index(drop=True)

    return shuffled_data, [shuffled_data[1].min(), shuffled_data[1].max()], "PlantDoc"


def load_and_prepare_data_Soybean_Dangerous_Insects(total_samples_to_check):

    # Dataset details
    dataset_name = "tarundalal/dangerous-insects-dataset"
    download_path = "./data/farm_insects"

    # Check if the dataset already exists
    if not os.path.exists(download_path):
        # Download the dataset with progress
        download_with_progress(dataset_name, path=download_path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
    
    base_directory =os.path.join(download_path,'farm_insects')
    
    expected_classes = [ 'Africanized Honey Bees', 'Aphids', 'Armyworms', 'Brown Marmorated Stink Bugs', 'Cabbage Loopers', 'Citrus Canker', 'Colorado Potato Beetles', 'Corn Borers', 'Corn Earworms', 'Fall Armyworms', 'Fruit Flies', 'Spider Mites', 'Thrips', 'Tomato Hornworms', 'Western Corn Rootworms' ]
    rename_folders(base_directory, expected_classes)

    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Dangerous Insects"


def load_and_prepare_data_IDC(total_samples_to_check):
    record_id = "12740714"
    # Create a directory to store the downloaded files
    download_path = "./data/IDC_data"
    
    # Store the current working directory
    original_dir = os.getcwd()
   
    # Get file URLs from the record metadata
    file_urls = get_file_urls(record_id)   
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        os.chdir(download_path)
         # Download files
        for url, filename in file_urls:
            download_file(url, filename)    
        
        os.chdir(original_dir)

        # Extract zip files named "images"
        for filename in os.listdir(download_path):
            if filename.lower() == "images.zip":
                zip_path = os.path.join(download_path, filename)
                extract_zip(zip_path, download_path)  
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")

    
    os.chdir(original_dir)
 

    # Return to the original directory
    base_directory = os.path.join(download_path,'images')
    labels_file = os.path.join(download_path,'class_label.xlsx')
    
    # Read the Excel file
    df = pd.read_excel(labels_file)    
    file_paths = []
    labels = []
    
    # Iterate through the dataframe
    for index, row in df.iterrows():
        plot_number = str(row['Plot#'])
        rating = row['Field Visual rating']
        # if after converting the rating to integer it is not a integer then skip it
        try:
            rating = int(rating)
        except:
            continue
        # Construct the filename
        filename = f"{plot_number}-p.jpg"
        file_path = os.path.join(base_directory, filename)
        
        # Check if the file exists
        if os.path.exists(file_path):
            file_paths.append(file_path)
            labels.append(rating)
    
    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Get unique labels
    expected_classes = sorted(data[1].unique())
    samples_per_class = int(total_samples_to_check / len(expected_classes))
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    shuffled_data = shuffle(sampled_data, random_state=random_state).reset_index(drop=True)
    
    print(f"Loaded {len(shuffled_data)} samples from {base_directory}")
    print(f"Label range: {shuffled_data[1].min()} to {shuffled_data[1].max()}")
    print(f"Samples per class: {samples_per_class}")
    
    return shuffled_data, expected_classes, "IDC"



def load_and_prepare_data_Soybean_PNAS(total_samples_to_check):

    record_id = "12747481"
    # Create a directory to store the downloaded files
    download_path = "./data/Soybean-PNAS"
    
    # Store the current working directory
    original_dir = os.getcwd()
   
    # Get file URLs from the record metadata
    file_urls = get_file_urls(record_id)   
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        os.chdir(download_path)
         # Download files
        for url, filename in file_urls:
            download_file(url, filename)  
        os.chdir(original_dir)      

        # Extract zip files named "images"
        for filename in os.listdir(download_path):
            if filename.lower() == "soybean_stress_identification.zip":
                zip_path = os.path.join(download_path, filename)
                extract_zip(zip_path, download_path)  
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")      
   

    base_directory = './data/Soybean-PNAS/Training Samples'

    expected_classes = ['Bacterial Blight','Bacterial Pustule','Frogeye Leaf Spot', 'Healthy' , 'Herbicide Injury' , 'Iron Deficiency Chlorosis', 'Potassium Deficiency','Septoria Brown Spot', 'Sudden Death Syndrome' ]
    
    soybean_stress_dict = {
    '0': 'Bacterial Blight',
    '1': 'Bacterial Pustule',
    '2': 'Frogeye Leaf Spot',
    '3': 'Healthy',
    '4': 'Herbicide Injury',
    '5': 'Iron Deficiency Chlorosis',
    '6': 'Potassium Deficiency',
    '7': 'Septoria Brown Spot',
    '8': 'Sudden Death Syndrome'
    }

    rename_folders_dict(base_directory, soybean_stress_dict )
    print(os.listdir(base_directory))
    samples_per_class = int(total_samples_to_check / len(expected_classes))
    file_paths = []
    labels = []

    for subdir in os.listdir(base_directory):
        if subdir == ".DS_Store":
            continue
        subdir_path = os.path.join(base_directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(subdir_path, filename))
                labels.append(subdir)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    sampled_data = pd.DataFrame(columns=[0, 1])
    for cls in expected_classes:
        class_data = data[data[1] == cls]
        if len(class_data) >= samples_per_class:
            class_sample = class_data.sample(n=samples_per_class, random_state=random_state)
        else:
            class_sample = class_data
            print(f"Warning: Not enough samples for class {cls}. Using all {len(class_data)} available samples.")
        sampled_data = pd.concat([sampled_data, class_sample], ignore_index=True)
    
    # Shuffle the sampled data
    print(f"Loaded {len(sampled_data)} samples from {base_directory}")
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Soybean Diseases"


def load_and_prepare_data_InsectCount(total_samples_to_check):
    record_id = "12747496"
    # Create a directory to store the downloaded files
    download_path = "./data/insectcount"
    
    # Store the current working directory
    original_dir = os.getcwd()
   
    # Get file URLs from the record metadata
    file_urls = get_file_urls(record_id)   
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        os.chdir(download_path)
         # Download files
        for url, filename in file_urls:
            download_file(url, filename)    

        os.chdir(original_dir)
        # Extract zip files named "images"
        for filename in os.listdir(download_path):
            if filename.lower() == "images.zip":
                zip_path = os.path.join(download_path, filename)
                extract_zip(zip_path, download_path)  

            if filename.lower() == "labels.zip":
                zip_path = os.path.join(download_path, filename)
                extract_zip(zip_path, download_path)  
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")

    
    os.chdir(original_dir)

    base_directory = './data/insectcount'
    images_dir = os.path.join(base_directory, 'images')
    labels_dir = os.path.join(base_directory, 'labels')
    
    file_paths = []
    labels = []

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = len(f.readlines())  # Count the number of rows
                file_paths.append(image_path)
                labels.append(label)

    data = pd.DataFrame({0: file_paths, 1: labels})
    
    # Use a fixed random state for deterministic sampling
    random_state = 42
    
    # Sample the data
    if len(data) > total_samples_to_check:
        sampled_data = data.sample(n=total_samples_to_check, random_state=random_state)
    else:
        sampled_data = data
        print(f"Warning: Not enough samples. Using all {len(data)} available samples.")
    
    # Shuffle the sampled data
    shuffled_data = shuffle(sampled_data, random_state=random_state).reset_index(drop=True)
    
    print(f"Loaded {len(shuffled_data)} samples from {base_directory}")
    print(f"Label range: {shuffled_data[1].min()} to {shuffled_data[1].max()}")
    return shuffled_data, [shuffled_data[1].min(), shuffled_data[1].max()], "InsectCount"