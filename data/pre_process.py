import pickle
import os
import torch
import numpy as np

def load_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def combine_pkl_files(data_dir='raw_data/'):
    combined_data = []

    # Iterate over each file in the data directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(data_dir, file_name)
            print(f"Loading data from {file_path}...")
            data = load_data_from_file(file_path)
            combined_data.extend(data)  # Add the data from this file to the combined list

    return combined_data

def normalize_tensor(tensor):
    # Compute mean and standard deviation
    mean = tensor.mean()
    std = tensor.std()
    
    # Apply normalization
    normalized_tensor = (tensor - mean) / (std + 1e-7)  # Adding epsilon to prevent division by zero
    return normalized_tensor


if __name__ == "__main__":
    combined_data = combine_pkl_files('raw_data/')
    
    print(f"Total number of tuples: {len(combined_data)}")
    
    normalized_dataset = [(normalize_tensor(tensor), label) for tensor, label in combined_data]

    # Save the normalized dataset
    with open('processed_data/all_data.pkl', 'wb') as f:
        pickle.dump(normalized_dataset, f)
