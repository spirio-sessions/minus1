import os
import sqlite3
import h5py
import pandas as pd


def encode_string(s):
    """
    Simple utility function to make sure a string is proper
    to be used in a SQLite query
    (different from PostgreSQL, no N to specify Unicode)
    EXAMPLE:
      That's my boy! -> 'That''s my boy!'
    """
    return "'" + s.replace("'", "''") + "'"


def explore_h5_group(group, indent=0):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print('  ' * indent + f"Group: {key}")
            explore_h5_group(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print('  ' * indent + f"Dataset: {key} - Shape: {item.shape} - Datatype: {item.dtype}")


def search_h5_files(root_dir):
    """Recursively search for .h5 files in the directory tree."""
    h5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files


def extract_jazz_songs(artist_term_db, track_metadata_db, h5_files):
    """Extracts and analyzes jazz songs from .h5 files."""
    # Connect to the artist_term.db database
    conn_artist_term = sqlite3.connect(artist_term_db)
    c_artist_term = conn_artist_term.cursor()

    # Connect to the track_metadata.db database
    conn_track_metadata = sqlite3.connect(track_metadata_db)
    c_track_metadata = conn_track_metadata.cursor()

    # Get artist IDs that have been tagged with 'jazz'
    q = "SELECT artist_id FROM artist_term WHERE term=" + encode_string('jazz')
    res = c_artist_term.execute(q)
    jazz_artists = [row[0] for row in res.fetchall()]
    print(f"* found {len(jazz_artists)} artists tagged with 'jazz'")

    jazz_songs = []

    # If there are jazz artists, fetch their songs from track_metadata.db
    if jazz_artists:
        jazz_artist_ids = ','.join([encode_string(artist_id) for artist_id in jazz_artists])
        q = f"SELECT * FROM songs WHERE artist_id IN ({jazz_artist_ids})"
        res = c_track_metadata.execute(q)
        track_metadata = res.fetchall()

        track_ids = [row[0] for row in track_metadata]

        # Iterate over .h5 files and extract relevant data for jazz songs
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                track_id = f['analysis']['songs']['track_id'][0].decode('utf-8')
                if track_id in track_ids:
                    artist_name = f['metadata']['songs']['artist_name'][0].decode('utf-8')
                    title = f['metadata']['songs']['title'][0].decode('utf-8')
                    tempo = f['analysis']['songs']['tempo'][0]
                    duration = f['analysis']['songs']['duration'][0]
                    year = f['musicbrainz']['songs']['year'][0]
                    jazz_songs.append([track_id, artist_name, title, tempo, duration, year])
                    print(f"Processed track: {track_id}, {title}, {artist_name}")

    # Close the cursor and the connection for artist_term.db
    c_artist_term.close()
    conn_artist_term.close()

    # Close the cursor and the connection for track_metadata.db
    c_track_metadata.close()

    return jazz_songs


def save_to_csv(data, output_csv):
    """Saves the extracted data to a CSV file."""
    columns = ['track_id', 'artist_name', 'title', 'tempo', 'duration', 'year']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f'Data saved to {output_csv}')


# Main script
if __name__ == "__main__":
    # Change these paths to your local configuration
    root_dir = 'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\millionsongdataset\millionsongsubset\MillionSongSubset'
    artist_term_db = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\millionsongdataset\\artist_term.db'
    track_metadata_db = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\millionsongdataset\\track_metadata.db'
    output_csv = 'jazz_songs.csv'

    h5_files = search_h5_files(root_dir)
    print(f"Found {len(h5_files)} .h5 files")

    jazz_songs = extract_jazz_songs(artist_term_db, track_metadata_db, h5_files)

    save_to_csv(jazz_songs, output_csv)
