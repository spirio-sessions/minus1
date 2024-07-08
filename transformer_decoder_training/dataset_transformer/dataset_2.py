import numpy as np
import torch
from torch.utils.data import Dataset


class AdvancedPianoDataset(Dataset):
    def __init__(self, data: list, seq_length: int, stride: int, start_token: np.ndarray):
        """
        :param data: A list of Songs. Each song is a list. A song has 2 numpy arrays: First the right hand then the left.
        :param seq_length: Length of sequence chunks.
        :param stride: How far the start of the second sequence should be from the first.
        :param start_token: The token to add at the beginning of each src and target chunk.
        """

        self.data = []
        for song in data:
            if len(song[0]) != len(song[1]):
                raise ValueError("Numpy arrays for Hands must have the same length")

            song = np.array(song)
            # Concatenate right and left hand (left hand first, index wise), so we combine right and left in one snapshot
            song = np.concatenate((song[1], song[0]), axis=1)

            if song.shape[1] != start_token.shape[1]:
                raise ValueError(f"Number of keys in snapshot must match sos token. song shape: {song.shape}, "
                                 f"SOS token shape: {start_token.shape}")

            self.data.append(song)

        self.seq_length = seq_length
        self.stride = stride
        self.start_token = start_token
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        for song in self.data:
            for i in range(0, len(song) - self.seq_length, self.stride):
                seq = song[i:i + self.seq_length]
                if len(seq) == self.seq_length:  # Ensure sequence is of required length
                    # Add the start token at the beginning of the sequence
                    seq_with_start = np.vstack((self.start_token, seq))
                    sequences.append(seq_with_start)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        return seq_tensor