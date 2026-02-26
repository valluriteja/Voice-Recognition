import os
import torch
from torch.utils.data import Dataset
from preprocess import extract_mel_spectrogram

# Only using 4 emotions to keep it simple and stable
emotion_map = {
    "01": 0,  # neutral
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
}

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []

        for actor in os.listdir(root_dir):
            actor_path = os.path.join(root_dir, actor)

            if not os.path.isdir(actor_path):
                continue

            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    emotion_code = file.split("-")[2]

                    if emotion_code in emotion_map:
                        self.file_paths.append(os.path.join(actor_path, file))
                        self.labels.append(emotion_map[emotion_code])

        print("Total samples:", len(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = extract_mel_spectrogram(self.file_paths[idx])

        # Add channel dimension for CNN
        mel = torch.tensor(mel).unsqueeze(0).float()

        label = torch.tensor(self.labels[idx]).long()

        return mel, label
