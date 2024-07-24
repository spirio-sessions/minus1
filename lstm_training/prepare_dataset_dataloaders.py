from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lstm_training.MelodyHarmonyDataset import MelodyHarmonyDataset


def prepare_dataset_dataloaders(data, seq_length, stride, batch_size, test_size=0.2, random_state=42):
    # Preparing data
    data_train, data_val = train_test_split(data, test_size=test_size, random_state=random_state)
    train_dataset = MelodyHarmonyDataset(data_train, seq_length, stride)
    val_dataset = MelodyHarmonyDataset(data_val, seq_length, stride)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader
