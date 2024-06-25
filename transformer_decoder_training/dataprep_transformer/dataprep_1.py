import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Function to add SOS and EOS tokens to each chunk
def add_sos_eos_to_chunks(chunks, sos_token):
    new_chunks = []
    for chunk in chunks:
        new_chunk = np.vstack([sos_token, chunk]) # eos token probably not neccessary
        # print(chunk.shape)
        # new_chunk = np.insert(chunk, 0, sos_token)
        new_chunks.append(new_chunk)
    return new_chunks


# Function to split sequences into chunks
def split_into_chunks(sequence, chunk_size):
    #print("sequence:", sequence.shape)
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]


# Function to filter out short chunks while maintaining pairs
def filter_short_chunks(chunks_1, chunks_2, min_length):
    filtered_chunks_1 = []
    filtered_chunks_2 = []
    for chunk_1, chunk_2 in zip(chunks_1, chunks_2):
        if len(chunk_1) >= min_length and len(chunk_2) >= min_length:
            filtered_chunks_1.append(chunk_1)
            filtered_chunks_2.append(chunk_2)
    return filtered_chunks_1, filtered_chunks_2


# Custom Dataset class
class PianoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# Prepare the dataset with paired sequences and SOS/EOS tokens for each chunk
def prepare_dataset(dataset_as_snapshots, chunk_size, min_length, sos_token):
    data = []
    for song in dataset_as_snapshots:
        track_1, track_2 = song
        assert len(track_1) == len(track_2), "Tracks must have the same length"

        chunks_1 = split_into_chunks(track_1, chunk_size)
        chunks_2 = split_into_chunks(track_2, chunk_size)
        chunks_1, chunks_2 = filter_short_chunks(chunks_1, chunks_2, min_length)

        # print("chunks diemsion:", chunks_1[0].size)

        # Add SOS and EOS tokens to each chunk
        chunks_1 = add_sos_eos_to_chunks(chunks_1, sos_token)
        chunks_2 = add_sos_eos_to_chunks(chunks_2, sos_token)

        for x, y in zip(chunks_1, chunks_2):
            data.append((x, y))
    return data