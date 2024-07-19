from data_preperation import dataset_snapshot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformer_decoder_training.dataset_transformer.dataset_2 import AdvancedPianoDataset

import numpy as np


def prepare_dataset_as_dataloaders(dataset_dir: str, snapshot_intervall: int, batch_size: int, seq_length: int,
                                   stride: int, test_size: int, sos_token: np.ndarray):
    # load data
    dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded(dataset_dir, snapshot_intervall)
    # filter snapshots to 88 piano notes
    dataset_as_snapshots = dataset_snapshot.filter_piano_range(dataset_as_snapshots)
    # reduce to 12 keys
    dataset_as_snapshots = dataset_snapshot.compress_existing_dataset_to_12keys(dataset_as_snapshots)

    # split songs into train, test and val
    train_data, temp_data = train_test_split(dataset_as_snapshots, test_size=test_size, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=True)

    # Create datasets
    train_dataset = AdvancedPianoDataset(train_data, seq_length, stride, sos_token)
    val_dataset = AdvancedPianoDataset(val_data, seq_length, stride, sos_token)
    test_dataset = AdvancedPianoDataset(test_data, seq_length, stride, sos_token)

    # Create loaders
    # Create DataLoaders for each subset with drop_last=True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader
