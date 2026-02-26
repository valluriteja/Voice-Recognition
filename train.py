import os
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ======================
# CONFIG
# ======================

DATA_PATH = "data/AudioWAV"   # change if needed
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 6
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LABEL MAPPING
# ======================

emotion_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

# ======================
# DATASET CLASS
# ======================

class CREMADDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.path, file_name)

        # Load audio
        signal, sr = librosa.load(file_path, sr=22050)

        # Mel Spectrogram
        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        spec = librosa.power_to_db(spec, ref=np.max)

        # Padding / Cutting
        if spec.shape[1] > MAX_LEN:
            spec = spec[:, :MAX_LEN]
        else:
            pad_width = MAX_LEN - spec.shape[1]
            spec = np.pad(spec, pad_width=((0,0),(0,pad_width)), mode='constant')

        spec = torch.tensor(spec).unsqueeze(0).float()

        # Extract label from filename
        emotion_code = file_name.split("_")[2]
        label = emotion_map[emotion_code]

        return spec, label

# ======================
# MODEL
# ======================

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 -> 16

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ======================
# LOAD DATA
# ======================

dataset = CREMADDataset(DATA_PATH)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# TRAINING SETUP
# ======================

model = EmotionCNN(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================
# TRAIN LOOP
# ======================

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

# ======================
# TEST ACCURACY
# ======================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# ======================
# SAVE MODEL
# ======================

torch.save(model.state_dict(), "emotion_cnn_50epochs.pth")
print("Model saved successfully!")