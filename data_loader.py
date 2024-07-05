import os
import pandas as pd
from sklearn.utils import shuffle

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


def l