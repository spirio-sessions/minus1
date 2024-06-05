import numpy as np
import pandas as pd


def load_csv_features(csv_path):
    df = pd.read_csv(csv_path)
    return df

def convert_features_to_snapshots(df, interval):
    snapshots = []
    for _, row in df.iterrows():
        duration = row['duration']
        num_snapshots = int(duration / interval)

        # Create feature dictionary
        features = {
            'track_id': row['track_id'],
            'artist_name': row['artist_name'],
            'title': row['title'],
            'tempo': row['tempo'],
            'duration': row['duration'],
            'year': row['year'],
            'key': row['key'],
            'mode': row['mode'],
            'time_signature': row['time_signature'],
            'bars_start': list(enumerate(row['bars_start'][:num_snapshots])),
            'beats_start': list(enumerate(row['beats_start'][:num_snapshots])),
            'segments_start': list(enumerate(row['segments_start'][:num_snapshots])),
            'segments_pitches': [list(enumerate(pitch)) for pitch in row['segments_pitches'][:num_snapshots]],
            'segments_timbre': [list(enumerate(timbre)) for timbre in row['segments_timbre'][:num_snapshots]],
            'sections_start': list(enumerate(row['sections_start'][:num_snapshots])),
            'tatums_start': list(enumerate(row['tatums_start'][:num_snapshots]))
        }

        snapshots.append(features)
    return snapshots


# Example usage
csv_path = 'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\millionsongdataset\jazz_songs.csv'
interval = 1.0  # 1 second intervals
df = load_csv_features(csv_path)
snapshots = convert_features_to_snapshots(df, interval)
print("test")
