import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, detection_confidence=0.8, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_hands=1,
        )

    def process(self, image):
        return self.hands.process(image)

    def get_landmark_coords(self, hand_landmarks, index):

        coords = np.array(
            [
                hand_landmarks.landmark[index].x,
                hand_landmarks.landmark[index].y,
                hand_landmarks.landmark[index].z,
            ]
        )

        return coords
