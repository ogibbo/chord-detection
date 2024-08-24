import cv2


class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def read_frame(self, auto_preprocess=False):
        ret, frame = self.cap.read()

        if auto_preprocess:
            frame = self.preprocess(frame)

        return ret, frame

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        return frame

    def postprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def show_frame(self, window_name, frame):
        cv2.imshow(window_name, frame)

    def wait_key(self, delay):
        return cv2.waitKey(delay)

    def shut_down(self):
        self.cap.release()
        cv2.destroyAllWindows()
