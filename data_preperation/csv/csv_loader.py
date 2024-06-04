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

    for filename, snapshots in filtered_dataset:
        base_filename = filename.split('\\')[1].split('.')[0]
        output = f"{output_dir}/{base_filename}.csv"
        snapshot_df = pd.DataFrame(snapshots)
        snapshot_df.to_csv(output, index=False)

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(file=base_filename)

    # Close the progress bar
    progress_bar.close()