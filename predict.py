import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

# 1. DEFINE MODEL (Must match training architecture)
class SER_CNN(nn.Module):
    def __init__(self):
        super(SER_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32); self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64); self.dropout = nn.Dropout(0.3); self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# 2. PREDICTION FUNCTION
def predict(file_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess Audio [cite: 238, 240]
    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Pad to 3 seconds
    target_len = int(22050 * 3.0)
    if len(y) > target_len: y = y[:target_len]
    else: y = np.pad(y, (0, target_len - len(y)), mode='constant')
    
    # Spectrogram
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512), ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    
    # Inference
    tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    model = SER_CNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Error: Could not load weights. Ensure 'ser_model.pth' is present.")
        return

    model.eval()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    
    emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    print(f"Prediction: {emotions[pred.item()]} ({conf.item()*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to .wav file")
    parser.add_argument('--weights', type=str, default='ser_model.pth', help="Path to model weights")
    args = parser.parse_args()
    predict(args.file, args.weights)
