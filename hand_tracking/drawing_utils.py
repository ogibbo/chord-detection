import mediapipe as mp


class DrawingUtils:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils

    def draw_landmarks(self, image, hand_landmarks, connections):
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            connections,
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(250, 44, 250), thickness=2, circle_radius=2
            ),
        )
