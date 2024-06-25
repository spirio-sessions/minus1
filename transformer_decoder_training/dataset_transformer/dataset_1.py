import torch
from torch.utils.data import DataLoader, Dataset


# Custom Dataset class
class PianoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
