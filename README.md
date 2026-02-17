# Speech Emotion Recognition (RAVDESS)

This project uses a Deep Learning model (2D CNN) to classify human speech into 8 emotions (Happy, Sad, Angry, etc.).

## Files
* `SER_Task.ipynb`: The complete training pipeline (Data loading, Preprocessing, CNN Architecture, Training, Evaluation).
* `predict.py`: A script to predict emotions on new audio files.

## How to Run
1. Install requirements: `pip install librosa torch numpy`
2. Run the prediction script:
   python predict.py --file "path/to/audio.wav"

## Results
* **Accuracy:** 55% 
