import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data_from_csv(directory):
    melody_data = []
    harmony_data = []

    melody_files = [f for f in os.listdir(directory) if f.endswith('_rightH.csv')]
    progress_bar = tqdm(total=len(melody_files))

    for melody_file in melody_files:
        base_filename = melody_file.replace('_rightH.csv', '')
        harmony_file = base_filename + '_leftH.csv'

        melody_df = pd.read_csv(os.path.join(directory, melody_file))
        harmony_df = pd.read_csv(os.path.join(directory, harmony_file))

        melody_data.append(melody_df.values)
        harmony_data.append(harmony_df.values)

        progress_bar.update(1)
        progress_bar.set_description(f"Loading dataset ({progress_bar.n}/{progress_bar.total}")

    return np.concatenate(melody_data), np.concatenate(harmony_data)
