from data_loader import load_and_prepare_data_YellowRust, load_and_prepare_data_FUSARIUM22
from data_loader import load_and_prepare_data_DiseaseQuantify

total_samples = 1000  # or any other number you prefer
data, classes, dataset_name = load_and_prepare_data_DiseaseQuantify (total_samples)