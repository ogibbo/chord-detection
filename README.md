# Visual Guitar Chord Detection

Used hand-tracking techniques to train a neural network to detect the guitar chord a user is playing. As a beginner myself, I found it took a while to learn the basic chords, but once I learned them, practising became more enjoyable because I could learn new songs quicker. I thought an app like this could accelerate this initial process.

![demo](media/app.gif)

## Project Overview

- **Hand Tracking:** Used Google’s MediaPipe computer vision framework to track hand movements and collect hand pose training data based on the chord being played.
- **Model Training:** Trained a neural network using PyTorch with PyTorch Lightning to predict the chord being played in real time.
- **Deployment:** Packaged the model into a containerized Streamlit application - ready to be deployed in the cloud (once I figure out how the docker container can access the host's webcam).

## Running the app

```bash
conda create --name chord_env
conda activate chord_env
pip install -r requirements.txt
streamlit run app.py
```

## Collecting more data to train a new model

For the demo I collected training data for a few chords, though only the final trained model is included in this repo. If you want to include more chords you can run main.py and initialise ChordDetection with collect_data=True and chord={your chord}. Once you've done this for all the new chords, you can process the training data by running pre_process.py before training the model with training.py

## References

- **MediaPipe Intro:** [link](https://www.youtube.com/watch?v=vQZ4IvB07ec)
- **PyTorch Lightning Intro:** [link](https://www.youtube.com/watch?v=NVxCKdp0NhQ)

