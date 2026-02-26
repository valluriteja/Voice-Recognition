import librosa
import numpy as np
import random

# ===== SETTINGS =====
SAMPLE_RATE = 16000
DURATION = 10                      # 🔥 10 seconds training
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# For 10 seconds at 16kHz with hop=512 → ~313 frames
FIXED_WIDTH = 313


def extract_mel_spectrogram(file_path, augment=True):
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # ===== Trim or Pad to 10 seconds =====
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    # ===== Data Augmentation (optional but recommended) =====
    if augment:
        # Add small noise
        if random.random() < 0.3:
            noise = 0.005 * np.random.randn(len(y))
            y = y + noise

        # Slight pitch shift
        if random.random() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-1, 1]))

        # Slight time stretch
        if random.random() < 0.3:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))

            # After time stretch, re-trim or pad again
            if len(y) > SAMPLES:
                y = y[:SAMPLES]
            else:
                y = np.pad(y, (0, SAMPLES - len(y)))

    # ===== Mel Spectrogram =====
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ===== Fix width to 313 =====
    if mel_db.shape[1] > FIXED_WIDTH:
        mel_db = mel_db[:, :FIXED_WIDTH]
    else:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))

    # ===== Normalize =====
    mean = np.mean(mel_db)
    std = np.std(mel_db)
    if std != 0:
        mel_db = (mel_db - mean) / std

    return mel_db