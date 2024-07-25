import numpy as np
import torch
from torch.utils.data import Dataset


class MelodyHarmonyDataset(Dataset):
    def __init__(self, data: list, seq_length: int, stride: int):
        """
        :param data: A list of Songs. Each song is a list. A song has 2 numpy arrays: First the right hand then the left.
        :param seq_length: Length of sequence chunks.
        :param stride: How far the start of the second sequence should be from the first.
        """

        self.data = []
        for song in data:
            assert len(song[0]) == len(song[1]), "Melody and harmony must be the same length"

            song = np.array(song)
            if song.shape[1] != 24:
                # Concatenate right and left hand (left hand first), so we combine right and left in one snapshot
                song = np.concatenate((song[1], song[0]), axis=1)

            self.data.append(song)

        self.seq_length = seq_length
        self.stride = stride
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        for song in self.data:
            for i in range(0, len(song) - self.seq_length, self.stride):
                seq = song[i:i + self.seq_length]
                if len(seq) == self.seq_length:  # Ensure sequence is of required length
                    sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        return seq_tensor
