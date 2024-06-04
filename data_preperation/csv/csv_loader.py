import os

import pandas as pd
from tqdm import tqdm


def export_melody_harmony_to_csv(melody_harmony_dataset, output_dir):
    for filename, melody_snapshots, harmony_snapshots in melody_harmony_dataset:
        # Prepare filenames for CSV
        base_filename = filename.split('/')[-1].split('.')[0]
        melody_filename = f"{output_dir}/{base_filename}_melody.csv"
        harmony_filename = f"{output_dir}/{base_filename}_harmony.csv"

        # Convert to DataFrames
        melody_df = pd.DataFrame(melody_snapshots)
        harmony_df = pd.DataFrame(harmony_snapshots)

        # Export to CSV
        melody_df.to_csv(melody_filename, index=False)
        harmony_df.to_csv(harmony_filename, index=False)


def export_maestro_hands_to_csv(filtered_dataset, output_dir):
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