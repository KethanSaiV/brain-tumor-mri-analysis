import os
import cv2
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, cls in enumerate(["healthy", "tumor"]):
            path = os.path.join(root, cls)
            for f in os.listdir(path):
                self.samples.append((os.path.join(path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = torch.tensor(img).unsqueeze(0).float()
        return img, label
