import cv2

from hand_tracking import (
    VideoStream,
    HandTracker,
    DrawingUtils,
    FingerTip,
    get_fret_string_dict,
)
from calibration import ManualCalibration, CHORDS


def main():

    # Initialising video stream, source may be different for you
    video_stream = VideoStream(source=0)
    hand_tracker = HandTracker()
    drawing_utils = DrawingUtils()

    manual_calibration = ManualCalibration()

    ftip_1 = FingerTip()
    ftip_2 = FingerTip()
    ftip_3 = FingerTip()

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

            # Extracting desired landmarks from hand tracking
            f1_landmark = hand_tracker.get_landmark_coords(hand_landmarks, 8)
            f2_landmark = hand_tracker.get_landmark_coords(hand_landmarks, 12)
            f3_landmark = hand_tracker.get_landmark_coords(hand_landmarks, 16)

            # Setting the coordinates for the fingertips
            ftip_1.set_coords(f1_landmark[0], f1_landmark[1])
            ftip_2.set_coords(f2_landmark[0], f2_landmark[1])
            ftip_3.set_coords(f3_landmark[0], f3_landmark[1])

        video_stream.show_frame("Hand Tracking", bgr_frame)

        if manual_calibration.complete:

            fret_string_dict = get_fret_string_dict(
                manual_calibration, ftip_1, ftip_2, ftip_3
            )

            # Check if fret string dict matches any dict in CHORDS
            for chord, chord_dict in CHORDS.items():
                if fret_string_dict == chord_dict:
                    print(f"Chord detected: {chord}")
                    break

        else:
            if video_stream.wait_key(100) & 0xFF == 32:
                manual_calibration.set_fretboard_coord(ftip_1)
            manual_calibration.check_if_set()

        if video_stream.wait_key(10) & 0xFF == ord("q"):
            break

    video_stream.shut_down()


if __name__ == "__main__":
    main()
