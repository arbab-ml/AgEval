import os
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
def load_and_prepare_data_SBRD(total_samples_to_check):
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/severity-based-rice-disease/train'
    expected_classes = ['Healthy', 'Mild Bacterial Blight', 'Mild Blast', 'Mild Brownspot', 'Mild Tungro', 'Severe Bacterial Blight', 'Severe Blast', 'Severe Brownspot', 'Severe Tungro']
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
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "SBRD"


def load_and_prepare_data_DurumWheat(total_samples_to_check):
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/Durum_Wheat_Dataset/Dataset2-Durum Wheat Video Images/processed'
    expected_classes = ['Foreign Matters', 'Starchy Kernels', 'Vitreous Kernels']
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
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Durum Wheat"


def load_and_prepare_data_soybean_seeds(total_samples_to_check):
    base_directory = 'data/Soybean Seeds'
    expected_classes = ['Broken', 'Immature', 'Intact', 'Skin-damaged', 'Spotted']
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

    base_directory = 'data/mango-leaf-disease-dataset'
    expected_classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
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
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/deepweeds'
    expected_classes = ['Chinee apple', 'Lantana', 'Negative', 'Snake weed', 'Siam weed', 'Prickly acacia', 'Parthenium', 'Rubber vine', 'Parkinsonia']
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
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/ip02-dataset/classification/train'
    classes_file = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/ip02-dataset/classes.txt'

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
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/bean-leaf-lesions-classification/train'
    expected_classes = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']
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
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/yellowrust19-yellow-rust-disease-in-wheat/YELLOW-RUST-19/YELLOW-RUST-19'
    expected_classes = ['Moderately Resistant (MR)', 'Moderately Susceptible (MS)', 'MRMS', 'No disease (0)', 'Resistant (R)', 'Susceptible (S)']
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
    return shuffle(sampled_data, random_state=random_state).reset_index(drop=True), expected_classes, "Yellow Rust 19"


def load_and_prepare_data_FUSARIUM22(total_samples_to_check):
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/FUSARIUM-22/dataset_raw'
    expected_classes = ['Highly Resistant', 'Highly Susceptible', 'Moderately Resistant', 'Resistant', 'Susceptible']
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




def load_and_prepare_data_InsectCount(total_samples_to_check):
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/insectcount/train'
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


def load_and_prepare_data_DiseaseQuantify(total_samples_to_check):
    base_directory = '/Users/muhammadarbabarshad/Downloads/AgEval-datasets/leaf-disease-quantification/data/data'
    images_dir = os.path.join(base_directory, 'images')
    masks_dir = os.path.join(base_directory, 'masks')
    
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