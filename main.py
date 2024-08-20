import torch
import pickle
import sys

from hand_tracking import VideoStream, HandTracker, DrawingUtils
from utils import get_hand_tensor, load_existing_model
from data import normalize_tensor, CHORD_LABELS_R

sys.path.append("models")


class ChordDetection:
    def __init__(
        self, model_fp, source=0, chord=None, collect_data=False, num_samples=1000
    ):
        self.chord = chord
        self.collect_data = collect_data
        self.num_samples = num_samples
        self.collected_data = []

        self.video_stream = VideoStream(source=source)
        self.hand_tracker = HandTracker()
        self.drawing_utils = DrawingUtils()

        self.predicted_chord = None

        if not self.collect_data:
            self.model = load_existing_model(model_fp)

    def process_frame(self):
        ret, frame = self.video_stream.read_frame(auto_preprocess=True)
        results = self.hand_tracker.process(frame)
        bgr_frame = self.video_stream.postprocess(frame)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.drawing_utils.draw_landmarks(
                    bgr_frame,
                    hand_landmarks,
                    self.hand_tracker.mp_hands.HAND_CONNECTIONS,
                )

                hand_tensor = get_hand_tensor(hand_landmarks)

                if self.collect_data:

                    if self.chord is None:
                        raise ValueError("Please provide a chord to collect data for")

                    self.collect_training_data(hand_tensor)
                else:
                    self.classify_chord(hand_tensor)

        return bgr_frame

    def collect_training_data(self, hand_tensor):
        self.collected_data.append((hand_tensor, self.chord))
        print(f"Collected {len(self.collected_data)} samples")

        if len(self.collected_data) >= self.num_samples:
            self.save_training_data()

    def save_training_data(self):
        with open(f"data/raw_data/tensors_{self.chord}.pkl", "wb") as handle:
            pickle.dump(self.collected_data, handle)
        print(
            f"Saved {len(self.collected_data)} samples to data/tensors_{self.chord}.pkl"
        )
        self.video_stream.shut_down()
        sys.exit()

    def classify_chord(self, hand_tensor):
        with torch.no_grad():
            normalized_hand_tensor = normalize_tensor(hand_tensor)
            flattened_tensor = normalized_hand_tensor.view(-1)
            batch_tensor = flattened_tensor.unsqueeze(0)

            output = self.model(batch_tensor)
            predicted_class = torch.argmax(output, dim=1)

            predicted_chord = CHORD_LABELS_R[predicted_class.item()]
            self.predicted_chord = predicted_chord

            print(f"Predicted Class: {predicted_chord}")

    def run(self):
        while self.video_stream.cap.isOpened():
            bgr_frame = self.process_frame()
            self.video_stream.show_frame("Hand Tracking", bgr_frame)

            if self.video_stream.wait_key(10) & 0xFF == ord("q"):
                break

        self.video_stream.shut_down()


if __name__ == "__main__":
    model_fp = "models/saved_models/model.pth"
    classifier = ChordDetection(model_fp)
    classifier.run()
