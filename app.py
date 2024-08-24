import streamlit as st
import cv2
import time
import random

from main import ChordDetection
from data import CHORD_LABELS

model_fp = "models/saved_models/model.pth"
classifier = ChordDetection(model_fp)

# Placeholders
stframe = st.empty()
target_chord_placeholder = st.empty()  # Placeholder for the target chord

# List of target chords
target_chords = list(CHORD_LABELS.keys())
target_chord = target_chords[0]

start_time = time.time()


def draw_predicted_chord(frame, chord, target_chord):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 10

    if chord == target_chord:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    cv2.circle(frame, (center_x, center_y), radius, color, thickness=10)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(chord, font, font_scale, thickness=4)[0]
    text_x = int(center_x - text_size[0] // 2)
    text_y = int(center_y + text_size[1] // 2)
    cv2.putText(frame, chord, (text_x, text_y), font, font_scale, color, thickness=4)


while classifier.video_stream.cap.isOpened():

    elapsed_time = time.time() - start_time
    if elapsed_time > 3:
        target_chord = random.choice(target_chords)
        start_time = time.time()  # Reset the timer

    target_chord_placeholder.markdown(
        f"<h2 style='text-align: center;'>Play the following chord: {target_chord}</h2>",
        unsafe_allow_html=True,
    )

    bgr_frame = classifier.process_frame()

    if classifier.predicted_chord:
        draw_predicted_chord(bgr_frame, classifier.predicted_chord, target_chord)

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    stframe.image(rgb_frame, channels="RGB")

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

classifier.video_stream.shut_down()
