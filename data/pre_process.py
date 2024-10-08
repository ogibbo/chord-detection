import pickle
import os

CHORD_LABELS = {
    "A": 0,
    "C": 1,
    "D": 2,
    "G": 3,
}

CHORD_LABELS_R = {v: k for k, v in CHORD_LABELS.items()}


def load_data_from_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def combine_pkl_files(data_dir="raw_data/"):
    combined_data = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Loading data from {file_path}...")
            data = load_data_from_file(file_path)
            combined_data.extend(
                data
            )

    return combined_data


def normalize_tensor(tensor):
    # Compute mean and standard deviation
    mean = tensor.mean()
    std = tensor.std()

    # Apply normalization
    normalized_tensor = (tensor - mean) / (
        std + 1e-7
    )  # Adding epsilon to prevent division by zero

    # changing origin to wrist
    normalized_tensor = normalized_tensor - normalized_tensor[0]

    return normalized_tensor


if __name__ == "__main__":
    combined_data = combine_pkl_files("raw_data/")

    print(f"Total number of tuples: {len(combined_data)}")

    normalized_dataset = [
        (normalize_tensor(tensor), CHORD_LABELS[label])
        for tensor, label in combined_data
    ]

    with open("processed_data/all_data.pkl", "wb") as f:
        pickle.dump(normalized_dataset, f)
