import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def export_maestro_hands_to_csv(filtered_dataset, output_dir):
    """
Export left and right hand snapshots from a filtered dataset to CSV files.

This function takes a filtered dataset containing pairs of left and right hand
snapshots, converts each pair to a CSV file, and saves them to a specified
output directory. Each pair of snapshots is saved as two separate CSV files
with filenames indicating the song number and hand (left or right).

Parameters:
filtered_dataset (list): A list of tuples, where each tuple contains two elements:
                         - left_hand_snapshots (array-like): Snapshots for the left hand.
                         - right_hand_snapshots (array-like): Snapshots for the right hand.
output_dir (str): The directory where the CSV files will be saved.

Returns:
None

Example:
>>> filtered_dataset = [([...], [...]), ([...], [...])]
>>> output_dir = './output_csvs'
>>> export_maestro_hands_to_csv(filtered_dataset, output_dir)
Exporting CSVs: 100%|███████████████████████████████████████████████| 2/2 [00:00<00:00, 10.00file/s]
"""
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the progress bar
    progress_bar = tqdm(total=len(filtered_dataset), desc="Exporting CSVs", unit="file", colour='#800080')

    for i, (left_hand_snapshots, right_hand_snapshots) in enumerate(filtered_dataset):
        # Create filenames for left and right hand CSVs
        base_filename = f"song_{i+1}"
        left_hand_output = f"{output_dir}/{base_filename}_leftH.csv"
        right_hand_output = f"{output_dir}/{base_filename}_rightH.csv"

        # Convert snapshots to DataFrames and save to CSVs
        left_hand_df = pd.DataFrame(left_hand_snapshots)
        right_hand_df = pd.DataFrame(right_hand_snapshots)
        left_hand_df.to_csv(left_hand_output, index=False)
        right_hand_df.to_csv(right_hand_output, index=False)

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(file=base_filename)

    # Close the progress bar
    progress_bar.close()


def export_maestro_hands_to_csv_transpose_to_each_key(filtered_dataset, output_dir):
    """
Export left and right hand snapshots from a filtered dataset to CSV files.

This function takes a filtered dataset containing pairs of left and right hand
snapshots, converts each pair to a CSV file, and saves them to a specified
output directory. Each pair of snapshots is saved as two separate CSV files
with filenames indicating the song number and hand (left or right).

Additionally, it transposes each song to 12 keys by incrementing each note.

Parameters:
filtered_dataset (list): A list of tuples, where each tuple contains two elements:
                         - left_hand_snapshots (array-like): Snapshots for the left hand.
                         - right_hand_snapshots (array-like): Snapshots for the right hand.
output_dir (str): The directory where the CSV files will be saved.

Returns:
None

Example:
>>> filtered_dataset = [([...], [...]), ([...], [...])]
>>> output_dir = './output_csvs'
>>> export_maestro_hands_to_csv(filtered_dataset, output_dir)
Exporting CSVs: 100%|███████████████████████████████████████████████| 2/2 [00:00<00:00, 10.00file/s]
"""
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the progress bar
    progress_bar = tqdm(total=len(filtered_dataset), desc="Exporting CSVs", unit="file", colour='#800080')

    for i, (left_hand_snapshots, right_hand_snapshots) in enumerate(filtered_dataset):
        # Convert snapshots to DataFrames
        left_hand_df = pd.DataFrame(left_hand_snapshots)
        right_hand_df = pd.DataFrame(right_hand_snapshots)

        # Transpose the data 12 times
        for j in range(12):
            transposed_left_hand_df = left_hand_df.copy()
            transposed_right_hand_df = right_hand_df.copy()

            # Shift the data
            for row in range(transposed_left_hand_df.shape[0]):
                transposed_left_hand_df.iloc[row] = np.roll(transposed_left_hand_df.iloc[row], j)
            for row in range(transposed_right_hand_df.shape[0]):
                transposed_right_hand_df.iloc[row] = np.roll(transposed_right_hand_df.iloc[row], j)

            # Create filenames for left and right hand CSVs
            base_filename = f"song_{i+1}_+{j}"
            left_hand_output = f"{output_dir}/{base_filename}_leftH.csv"
            right_hand_output = f"{output_dir}/{base_filename}_rightH.csv"

            # Save transposed DataFrames to CSVs
            transposed_left_hand_df.to_csv(left_hand_output, index=False, header=False)
            transposed_right_hand_df.to_csv(right_hand_output, index=False, header=False)

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(file=f"song_{i+1}")

    # Close the progress bar
    progress_bar.close()
