import torch
from torch.utils.data import Dataset


class MelodyHarmonyDataset(Dataset):
    """
A custom PyTorch Dataset for melody and harmony data.

This class handles the loading and indexing of melody and harmony datasets,
providing an interface to retrieve individual samples in a format suitable
for training models.

Attributes:
melody (array-like): The melody data, where each element is a sequence of features.
harmony (array-like): The harmony data, where each element is a corresponding sequence of features.

Methods:
__len__():
    Returns the number of samples in the dataset.
__getitem__(idx):
    Retrieves the sample (melody and harmony pair) at the specified index.

Example:
>>> melody_data = [[...], [...], ...]
>>> harmony_data = [[...], [...], ...]
>>> dataset = MelodyHarmonyDataset(melody_data, harmony_data)
>>> print(len(dataset))
100
>>> melody, harmony = dataset[0]
>>> print(melody.shape, harmony.shape)
torch.Size([1, sequence_length]) torch.Size([sequence_length])
"""
    def __init__(self, melody, harmony):
        """
        Initializes the MelodyHarmonyDataset with melody and harmony data.

        Parameters:
        melody (array-like): The melody data, where each element is a sequence of features.
        harmony (array-like): The harmony data, where each element is a corresponding sequence of features.
        """
        self.melody = melody
        self.harmony = harmony

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        int: The number of samples in the dataset.
        """
        return len(self.melody)

    def __getitem__(self, idx):
        """
        Retrieves the sample (melody and harmony pair) at the specified index.

        Parameters:
        idx (int): The index of the sample to retrieve.

        Returns:
        tuple: A tuple containing:
            - melody (torch.Tensor): The melody data at the specified index, with an added sequence_length dimension.
            - harmony (torch.Tensor): The harmony data at the specified index.
        """
        return (
            torch.tensor(self.melody[idx], dtype=torch.float32).unsqueeze(0),  # Add sequence_length dimension
            torch.tensor(self.harmony[idx], dtype=torch.float32)
        )
