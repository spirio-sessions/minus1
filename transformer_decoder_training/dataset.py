import torch
from torch.utils.data import Dataset, DataLoader


class PianoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song = self.data[idx]
        right_hand = song[0]
        left_hand = song[1]
        return torch.tensor(right_hand, dtype=torch.float32), torch.tensor(left_hand, dtype=torch.float32)


def create_dataloader(data, batch_size=32, shuffle=True):
    dataset = PianoDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
