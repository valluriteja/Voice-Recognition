import torch
import torch.nn.functional as F
import numpy as np
import librosa

from model import EmotionCNN

# ===== SETTINGS (must match training) =====
SAMPLE_RATE = 16000
DURATION = 10
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FIXED_WIDTH = 94

MODEL_PATH = r"D:\3'2\VoiceEmotionProject\emotion_model.pth"

# Emotion labels (RAVDESS order)
EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]


def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] > FIXED_WIDTH:
        mel_db = mel_db[:, :FIXED_WIDTH]
    else:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))

    # normalization (same as training)
    mean = np.mean(mel_db)
    std = np.std(mel_db)
    if std != 0:
        mel_db = (mel_db - mean) / std

    return mel_db


def predict(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    mel = extract_mel_spectrogram(file_path)

    mel_tensor = torch.tensor(mel, dtype=torch.float32)
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mel_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print("\nPrediction:")
    print("Emotion:", EMOTIONS[predicted_class])
    print("Confidence:", probabilities[0][predicted_class].item())


if __name__ == "__main__":
    test_file = r"D:\3'2\VoiceEmotionProject\RAVDESS\Actor_01\03-01-05-01-01-01-01.wav"
    predict(test_file)
