import torch
from torch.utils.data import Dataset


class MelodyHarmonyDataset(Dataset):
    def __init__(self, melody, harmony):
        self.melody = melody
        self.harmony = harmony

    def __len__(self):
        return len(self.melody)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.melody[idx], dtype=torch.float32).unsqueeze(0),  # Add sequence_length dimension
            torch.tensor(self.harmony[idx], dtype=torch.float32)
        )
