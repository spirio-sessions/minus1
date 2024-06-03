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

# Alle Sequenzlängen in einem Batch müssen die gleiche länge haben. kürzere werden gepadded
def pad_collate(batch):
    (melodies, accompaniments) = zip(*batch)

    # Find max length
    max_len = max([melody.shape[0] for melody in melodies])

    # Pad sequences
    padded_melodies = []
    padded_accompaniments = []
    for melody, accompaniment in zip(melodies, accompaniments):
        pad_len_melody = max_len - melody.shape[0]
        pad_len_accompaniment = max_len - accompaniment.shape[0]

        padded_melodies.append(torch.cat([melody, torch.zeros((pad_len_melody, 88))], dim=0))
        padded_accompaniments.append(torch.cat([accompaniment, torch.zeros((pad_len_accompaniment, 88))], dim=0))

    return torch.stack(padded_melodies), torch.stack(padded_accompaniments)


def create_dataloader(data, batch_size=32, shuffle=True):
    dataset = PianoDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    return dataloader
