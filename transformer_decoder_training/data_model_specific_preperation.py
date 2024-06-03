import numpy as np


def split_sequences(data, max_len=1000):
    split_data = []
    for song in data:
        right_hand, left_hand = song
        num_splits = int(np.ceil(len(right_hand) / max_len))
        for i in range(num_splits):
            start_idx = i * max_len
            end_idx = min((i + 1) * max_len, len(right_hand))
            split_data.append((right_hand[start_idx:end_idx], left_hand[start_idx:end_idx]))
    return split_data