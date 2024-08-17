import mediapipe as mp


class HandTracker:
    def __init__(self, detection_confidence=0.8, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, image):
        # Process the image to detect hand landmarks
        return self.hands.process(image)
