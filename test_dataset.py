from dataset import RAVDESSDataset
from torch.utils.data import DataLoader

dataset = RAVDESSDataset("RAVDESS")

print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for mel, label in loader:
    print("Batch shape:", mel.shape)
    print("Labels:", label)
    break
