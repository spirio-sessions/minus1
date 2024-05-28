import os
import numpy as np
import pandas as pd


def load_data_from_csv(directory):
    melody_data = []
    harmony_data = []

    melody_files = [f for f in os.listdir(directory) if f.endswith('_rightH.csv')]
    for melody_file in melody_files:
        base_filename = melody_file.replace('_rightH.csv', '')
        harmony_file = base_filename + '_leftH.csv'

        melody_df = pd.read_csv(os.path.join(directory, melody_file))
        harmony_df = pd.read_csv(os.path.join(directory, harmony_file))

        melody_data.append(melody_df.values)
        harmony_data.append(harmony_df.values)

    return np.concatenate(melody_data), np.concatenate(harmony_data)
