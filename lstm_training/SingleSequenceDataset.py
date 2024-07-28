import torch
from torch.utils.data import Dataset

class SingleSequenceDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1  # One single song

    def __getitem__(self, idx):
        item_tensor = torch.tensor(self.data, dtype=torch.float32)
        return item_tensor
