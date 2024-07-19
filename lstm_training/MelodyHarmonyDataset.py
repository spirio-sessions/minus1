import torch
from torch.utils.data import Dataset


class MelodyHarmonyDataset(Dataset):

    def __init__(self, melody, harmony):
        """
        Initializes the MelodyHarmonyDataset with melody and harmony data.

        Parameters:
        melody (array-like): The melody data, where each element is a sequence of features.
        harmony (array-like): The harmony data, where each element is a corresponding sequence of features.
        """
        self.data = [torch.cat((torch.tensor(m, dtype=torch.float32), torch.tensor(h, dtype=torch.float32)), dim=0) for m, h in zip(melody, harmony)]


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        int: The number of samples in the dataset.
        """
        return len(self.data)

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
        return self.data[idx]
