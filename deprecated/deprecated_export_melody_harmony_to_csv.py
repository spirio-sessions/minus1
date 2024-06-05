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
