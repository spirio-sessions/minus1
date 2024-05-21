import pandas as pd
import numpy as np


def load_csv_features(csv_path):
    df = pd.read_csv(csv_path)
    return df


def convert_features_to_snapshots(df, interval):
    snapshots = []
    for _, row in df.iterrows():
        # Example conversion, here we will use duration and tempo to simulate intervals
        duration = row['duration']
        tempo = row['tempo']
        num_snapshots = int(duration / interval)

        # Simulate active notes based on some features (this is an arbitrary example)
        active_notes = [0] * 128
        active_notes[int(tempo) % 128] = 1  # Simplified example

        track_snapshots = []
        for _ in range(num_snapshots):
            track_snapshots.append(active_notes[:])

        snapshots.append(track_snapshots)
    return snapshots


# Example usage
csv_path = 'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\millionsongdataset\jazz_songs.csv'
interval = 1.0  # 1 second intervals
df = load_csv_features(csv_path)
snapshots = convert_features_to_snapshots(df, interval)
print("test")
