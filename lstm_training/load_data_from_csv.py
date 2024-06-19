import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data_from_csv(directory, data_cap=0):
    """
    Load melody and harmony data from CSV files in a specified directory.

    This function scans a directory for pairs of CSV files containing melody
    and harmony data. Melody files are identified by the suffix '_rightH.csv',
    and harmony files by the corresponding '_leftH.csv'. It reads the data
    from these files into Pandas DataFrames, converts them to NumPy arrays,
    and returns the concatenated arrays of melody and harmony data.

    Parameters:
    directory (str): The path to the directory containing the CSV files.

    Returns:
    tuple: A tuple containing two NumPy arrays:
        - The first array contains concatenated melody data.
        - The second array contains concatenated harmony data.

    Raises:
    FileNotFoundError: If a corresponding harmony file is not found for a melody file.
    pd.errors.EmptyDataError: If a CSV file is empty.
    pd.errors.ParserError: If there is a parsing error while reading a CSV file.

    Example:
    >>> melody_data, harmony_data = load_data_from_csv('/path/to/directory')
    >>> print(melody_data.shape)
    (num_samples, num_features)
    >>> print(harmony_data.shape)
    (num_samples, num_features)
    """
    melody_data = []
    harmony_data = []

    melody_files = [f for f in os.listdir(directory) if f.endswith('_rightH.csv')]

    # Checks if a cap is wanted and limits the input
    if data_cap != 0:
        melody_files = melody_files[:data_cap]
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
