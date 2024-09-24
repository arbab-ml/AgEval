# import pandas as pd
# import numpy as np
# import os

# # Read the CSV file
# df = pd.read_csv('/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib/results/GPT-4o/DeepWeeds.csv')

# # Function to calculate accuracy
# def calc_accuracy(true_labels, predicted_labels):
#     return (true_labels == predicted_labels).mean()

# # Print basic information about the dataset
# print(f"Total samples: {len(df)}")
# print(f"Columns: {df.columns.tolist()}")

# # Create a results folder if it doesn't exist
# results_folder = 'results_analysis'
# os.makedirs(results_folder, exist_ok=True)

# # Initialize a list to store results
# results = []

# # Calculate accuracy for all shot counts
# shot_counts = [0, 1, 2, 4, 8]
# for shots in shot_counts:
#     for method in ['Embedding', 'Random']:
#         column_name = f"{method} # of Shots {shots}"
        
#         # Check for any mismatched predictions
#         mismatched_count = (df['1'] != df[column_name]).sum()
        
#         # Calculate accuracy
#         accuracy = calc_accuracy(df['1'], df[column_name])
        
#         # Calculate average same category count
#         same_category_count = df[f"{method} Same Category Count {shots}"].mean()
        
#         # Store results
#         results.append({
#             'Shots': shots,
#             'Method': method,
#             'Mismatched_Count': mismatched_count,
#             'Accuracy': accuracy,
#             'Avg_Same_Category_Count': same_category_count
#         })

# # Create a DataFrame from the results
# results_df = pd.DataFrame(results)

# # Save results to CSV
# results_df.to_csv(os.path.join(results_folder, 'accuracy_results.csv'), index=False)

# # Save distribution of true labels
# true_label_distribution = df['1'].value_counts(normalize=True)
# true_label_distribution.to_csv(os.path.join(results_folder, 'true_label_distribution.csv'))

# # Save missing values information
# missing_values = df.isnull().sum()
# missing_values.to_csv(os.path.join(results_folder, 'missing_values.csv'))

# print(f"Results saved in the '{results_folder}' folder.")

# # Additional analysis
# print("\nDistribution of true labels:")
# print(df['1'].value_counts(normalize=True))

# # Check for any missing values
# print("\nMissing values:")
# print(df.isnull().sum())

# # Print a few rows for manual verification
# print("\nSample rows:")
# print(df.head())

# # Print unique values in prediction columns
# for shots in shot_counts:
#     for method in ['Embedding', 'Random']:
#         column_name = f"{method} # of Shots {shots}"
#         print(f"\nUnique values in {column_name}:")
#         print(df[column_name].unique())