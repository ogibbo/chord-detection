# Chord Detection

Used hand-tracking techniques to train a neural network to detect the guitar chord a user is playing. As a beginner myself, I found it challenging to learn the basic chords, but once I learned them, practising became more enjoyable because I could learn new songs quicker. I thought an app like this could accelerate that process.

## Project Overview

- **Hand Tracking:** Used Google’s MediaPipe computer vision framework to track hand movements and collect training data for chord detection.
- **Model Training:** Trained a neural network using PyTorch with PyTorch Lightning to recognize different guitar chords based on hand positions.
- **Deployment:** Packaged the model into a containerized Streamlit application for easy access and use - ready to be deployed in the cloud.

