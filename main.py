import cv2

from hand_tracking import VideoStream, HandTracker, DrawingUtils
from calibration import ManualCalibration


def main():

    # Initialising video stream, source may be different for you
    video_stream = VideoStream(source=0)
    hand_tracker = HandTracker()
    drawing_utils = DrawingUtils()

    manual_calibration = ManualCalibration()

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

            first_finger = hand_tracker.get_landmark_coords(hand_landmarks, 8)
            second_finger = hand_tracker.get_landmark_coords(hand_landmarks, 12)
            third_finger = hand_tracker.get_landmark_coords(hand_landmarks, 16)

        video_stream.show_frame("Hand Tracking", bgr_frame)

        if manual_calibration.complete:
            fret_number, string_number = manual_calibration.get_fret_and_string(
                first_finger[0], first_finger[1]
            )
            print(f"Fret: {fret_number}, String: {string_number}")
        else:
            if video_stream.wait_key(100) & 0xFF == 32:
                manual_calibration.set_fretboard_coord(first_finger)
            manual_calibration.check_if_set()

        if video_stream.wait_key(10) & 0xFF == ord("q"):
            break

    video_stream.shut_down()


if __name__ == "__main__":
    main()
