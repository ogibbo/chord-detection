import cv2
import numpy as np
import torch
import pickle

from hand_tracking import (
    VideoStream,
    HandTracker,
    DrawingUtils,
)
from utils import get_hand_tensor

def main():

    # To do: Figure out better way to handle this
    GET_TRAINING_DATA = True
    CHORD = "C"
    COLLECTED_DATA = []
    N = 1000

    # Initialising video stream, source may be different for you
    video_stream = VideoStream(source=0)
    hand_tracker = HandTracker()
    drawing_utils = DrawingUtils()


    while video_stream.cap.isOpened():

        ret, frame = video_stream.read_frame(auto_preprocess=True)
        results = hand_tracker.process(frame)
        bgr_frame = video_stream.postprocess(frame)

        # Draw the landmarks on the frame
        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                drawing_utils.draw_landmarks(
                    bgr_frame, hand_landmarks, hand_tracker.mp_hands.HAND_CONNECTIONS
                )

                ######### Add functionality to get training data #########
                # Get the landmarks of the hand and save them as a 
                # hand_tensor = get_hand_tensor(hand_landmarks)

                # COLLECTED_DATA.append((hand_tensor, CHORD))
                # video_stream.wait_key(10)
                # print(len(COLLECTED_DATA))
                # if len(COLLECTED_DATA) == N:
                #     with open(f'data/tensors_{CHORD}.pkl', 'wb') as handle:
                #         pickle.dump(COLLECTED_DATA, handle)
                #     break


        video_stream.show_frame("Hand Tracking", bgr_frame)
        
        if video_stream.wait_key(10) & 0xFF == ord("q"):
            break

    video_stream.shut_down()


if __name__ == "__main__":
    main()
