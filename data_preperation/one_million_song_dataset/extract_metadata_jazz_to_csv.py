import os
import sqlite3
import h5py
import pandas as pd
import numpy as np

def encode_string(s):
    return "'" + s.replace("'", "''") + "'"

def search_h5_files(root_dir):
    h5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def extract_jazz_songs(artist_term_db, track_metadata_db, h5_files):
    conn_artist_term = sqlite3.connect(artist_term_db)
    c_artist_term = conn_artist_term.cursor()

    conn_track_metadata = sqlite3.connect(track_metadata_db)
    c_track_metadata = conn_track_metadata.cursor()

    q = "SELECT artist_id FROM artist_term WHERE term=" + encode_string('jazz')
    res = c_artist_term.execute(q)
    jazz_artists = [row[0] for row in res.fetchall()]
    print(f"* found {len(jazz_artists)} artists tagged with 'jazz'")

    jazz_songs = []

    if jazz_artists:
        jazz_artist_ids = ','.join([encode_string(artist_id) for artist_id in jazz_artists])
        q = f"SELECT * FROM songs WHERE artist_id IN ({jazz_artist_ids})"
        res = c_track_metadata.execute(q)
        track_metadata = res.fetchall()

        track_ids = [row[0] for row in track_metadata]

        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                track_id = f['analysis']['songs']['track_id'][0].decode('utf-8')
                if track_id in track_ids:
                    artist_name = f['metadata']['songs']['artist_name'][0].decode('utf-8')
                    title = f['metadata']['songs']['title'][0].decode('utf-8')
                    tempo = f['analysis']['songs']['tempo'][0]
                    duration = f['analysis']['songs']['duration'][0]
                    year = f['musicbrainz']['songs']['year'][0]

                    bars_start = f['analysis']['bars_start'][:]
                    beats_start = f['analysis']['beats_start'][:]
                    segments_start = f['analysis']['segments_start'][:]
                    segments_pitches = f['analysis']['segments_pitches'][:]
                    segments_timbre = f['analysis']['segments_timbre'][:]
                    sections_start = f['analysis']['sections_start'][:]
                    tatums_start = f['analysis']['tatums_start'][:]
                    key = f['analysis']['songs']['key'][0]
                    mode = f['analysis']['songs']['mode'][0]
                    time_signature = f['analysis']['songs']['time_signature'][0]

                    features = {
                        'track_id': track_id,
                        'artist_name': artist_name,
                        'title': title,
                        'tempo': tempo,
                        'duration': duration,
                        'year': year,
                        'key': key,
                        'mode': mode,
                        'time_signature': time_signature,
                        'bars_start': bars_start if len(bars_start) > 0 else [],
                        'beats_start': beats_start if len(beats_start) > 0 else [],
                        'segments_start': segments_start if len(segments_start) > 0 else [],
                        'segments_pitches': segments_pitches if len(segments_pitches) > 0 else [],
                        'segments_timbre': segments_timbre if len(segments_timbre) > 0 else [],
                        'sections_start': sections_start if len(sections_start) > 0 else [],
                        'tatums_start': tatums_start if len(tatums_start) > 0 else []
                    }

                    jazz_songs.append(features)
                    print(f"Processed track: {track_id}, {title}, {artist_name}")

    c_artist_term.close()
    conn_artist_term.close()
    c_track_metadata.close()
    conn_track_metadata.close()

    return pd.DataFrame(jazz_songs)


# Main script
if __name__ == "__main__":
    root_dir = 'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\millionsongdataset\millionsongsubset\MillionSongSubset'
    artist_term_db = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\millionsongdataset\\artist_term.db'
    track_metadata_db = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\millionsongdataset\\track_metadata.db'
    output_csv = 'jazz_songs.csv'

    h5_files = search_h5_files(root_dir)
    print(f"Found {len(h5_files)} .h5 files")

    jazz_songs_df = extract_jazz_songs(artist_term_db, track_metadata_db, h5_files)

    jazz_songs_df.to_csv(output_csv, index=False)
    print(f'Data saved to {output_csv}')
