import numpy as np


def filter_piano_range(dataset_as_snapshots):
    filtered_dataset = []

    for left_hand_snapshots, right_hand_snapshots in dataset_as_snapshots:
        filtered_left_hand_snapshots = [snapshot[21:109] for snapshot in left_hand_snapshots]
        filtered_right_hand_snapshots = [snapshot[21:109] for snapshot in right_hand_snapshots]
        filtered_dataset.append((np.array(filtered_left_hand_snapshots), np.array(filtered_right_hand_snapshots)))

    return filtered_dataset
