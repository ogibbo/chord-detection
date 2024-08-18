import pickle
import os

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

if __name__ == "__main__":
    combined_data = combine_pkl_files('raw_data/')
    print(f"Total number of tuples: {len(combined_data)}")
    
    
