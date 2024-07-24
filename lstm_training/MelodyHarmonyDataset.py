import numpy as np
import torch
from torch.utils.data import Dataset


class MelodyHarmonyDataset(Dataset):
    def __init__(self, melody: np.ndarray, harmony: np.ndarray, seq_length: int, stride: int):
        assert len(melody) == len(harmony), "Melody and harmony must be the same length"
        self.melody = melody
        self.harmony = harmony
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # Calculate the number of sequences we can extract from the data
        return len(self.melody) - self.seq_length

    def __getitem__(self, idx):
        # Felix Code:
        # transformer_decoder_training -> dataset_transformer -> dataset_2.py
        # for song in data:
        # song = np.concatenate((song[1], song[0]), axis=1)
        # self.data.append(song)
        # Extract input and target sequences
        input_start = idx
        input_end = input_start + self.seq_length
        target_start = input_start + 1
        target_end = target_start + self.seq_length

        input_segment = np.concatenate((self.melody[input_start:input_end], self.harmony[input_start:input_end]), axis=1)
        target_segment = self.harmony[target_start:target_end]

        return torch.tensor(input_segment, dtype=torch.float32), torch.tensor(target_segment, dtype=torch.long)




