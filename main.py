import cv2

from hand_tracking import (
    VideoStream,
    HandTracker,
    DrawingUtils,
)

def main():

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

        video_stream.show_frame("Hand Tracking", bgr_frame)
        
        if video_stream.wait_key(10) & 0xFF == ord("q"):
            break

    video_stream.shut_down()


if __name__ == "__main__":
    main()
