import os
import pandas as pd
from tqdm import tqdm


def load_data_from_csv(directory, data_cap=0):
    """
    Load melody and harmony data from CSV files in a specified directory.

    This function scans a directory for pairs of CSV files containing melody
    and harmony data. Melody files are identified by the suffix '_rightH.csv',
    and harmony files by the corresponding '_leftH.csv'. It reads the data
    from these files into Pandas DataFrames, converts them to NumPy arrays,
    and returns a list of songs. Each song is represented as a list containing
    two NumPy arrays: one for the melody (right hand) and one for the harmony (left hand).

    Parameters:
    directory (str): The path to the directory containing the CSV files.
    data_cap (int): Optional cap to limit the number of files processed.

    Returns:
    list: A list where each element is a list containing two NumPy arrays:
        - The first array contains the melody data.
        - The second array contains the harmony data.

    Raises:
    FileNotFoundError: If a corresponding harmony file is not found for a melody file.
    pd.errors.EmptyDataError: If a CSV file is empty.
    pd.errors.ParserError: If there is a parsing error while reading a CSV file.

    Example:
    >>> songs = load_data_from_csv('/path/to/directory')
    >>> print(len(songs))
    >>> print(songs[0][0].shape)  # Shape of the first song's melody data
    >>> print(songs[0][1].shape)  # Shape of the first song's harmony data
    """
    data = []

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

        song = [melody_df.values, harmony_df.values]
        data.append(song)

        progress_bar.update(1)
        progress_bar.set_description(f"Loading dataset ({progress_bar.n}/{progress_bar.total})")

    return data
