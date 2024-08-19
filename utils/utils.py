import numpy as np
import torch


def get_hand_tensor(hand_landmarks):

    landmark_list = []

    for landmark in hand_landmarks.landmark:

        landmark_list.append([landmark.x, landmark.y, landmark.z])
        landmark_array = np.array(landmark_list)
        landmark_tensor = torch.tensor(landmark_array, dtype=torch.float32)
    return landmark_tensor


def load_existing_model(path):

    with open(path, "rb") as f:
        model = torch.load(f)
        model.eval()

    return model
