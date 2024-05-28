import mido

import os
import fnmatch

import numpy as np
import pandas as pd

# for progress bar:
from tqdm import tqdm

# for multithread:
import concurrent.futures


# interval in seconds
# function will process all events happening in an intervall
# including intervall start, excluding intervall end
# snapshot will show all active notes at the end of the intervall
def snapshot_active_notes_from_midi(file_path, interval):
    mid = mido.MidiFile(file_path)
    snapshots_track_1 = []
    snapshots_track_2 = []
    active_notes_track_1 = [0] * 128
    active_notes_track_2 = [0] * 128
    current_time = 0
    snapshot_time = 0

    track_1_index = 0
    track_2_index = 1

    for msg in mid:
        current_time += msg.time

        while current_time >= snapshot_time + interval:
            if track_1_index < len(mid.tracks):
                snapshots_track_1.append(active_notes_track_1[:])
            if track_2_index < len(mid.tracks):
                snapshots_track_2.append(active_notes_track_2[:])
            snapshot_time += interval

        if msg.type == 'note_on':
            if msg.velocity == 0:
                if msg.channel == track_1_index:
                    active_notes_track_1[msg.note] = 0
                elif msg.channel == track_2_index:
                    active_notes_track_2[msg.note] = 0
            else:
                if msg.channel == track_1_index:
                    active_notes_track_1[msg.note] = 1
                elif msg.channel == track_2_index:
                    active_notes_track_2[msg.note] = 1
        elif msg.type == 'note_off':
            if msg.channel == track_1_index:
                active_notes_track_1[msg.note] = 0
            elif msg.channel == track_2_index:
                active_notes_track_2[msg.note] = 0

    snapshots_track_1 = np.array(snapshots_track_1)
    snapshots_track_2 = np.array(snapshots_track_2)

    return snapshots_track_1, snapshots_track_2


def find_midi_files(root_dir, pattern=None):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename.lower(), '*.midi') or fnmatch.fnmatch(filename.lower(), '*.mid'):
                if pattern is None or fnmatch.fnmatch(filename.lower(), f'*{pattern.lower()}*'):
                    midi_files.append(os.path.join(dirpath, filename))
    return midi_files


def process_dataset(dataset_dir, interval, pattern=None):
    midi_files = find_midi_files(dataset_dir, pattern)

    files_as_snapshots = []

    # Initialize tqdm with the number of MIDI files
    progress_bar = tqdm(total=len(midi_files))

    for midi_file in midi_files:
        snapshots_array_1, snapshots_array_2 = snapshot_active_notes_from_midi(midi_file, interval)
        files_as_snapshots.append((midi_file, snapshots_array_1, snapshots_array_2))
        progress_bar.update(1)
        progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

    # Close the progress bar
    progress_bar.close()

    return files_as_snapshots


def __process_single_midi(midi_file, interval):
    snapshots_array_1, snapshots_array_2 = snapshot_active_notes_from_midi(midi_file, interval)
    return midi_file, snapshots_array_1, snapshots_array_2


def process_dataset_multithreaded(dataset_dir, interval, pattern=None):
    midi_files = find_midi_files(dataset_dir, pattern)

    files_as_snapshots = []

    # Split MIDI files into chunks for processing
    num_chunks = min(len(midi_files), os.cpu_count() or 1)  # Use CPU count as a default if os.cpu_count() returns None
    midi_file_chunks = [midi_files[i::num_chunks] for i in range(num_chunks)]

    # Initialize tqdm with the total number of MIDI files
    progress_bar = tqdm(total=len(midi_files))

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(__process_single_midi, midi_file, interval) for chunk in midi_file_chunks for
                   midi_file in chunk]
        for future in concurrent.futures.as_completed(futures):
            files_as_snapshots.append(future.result())
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

    # Close the progress bar
    progress_bar.close()

    print(f"processed {len(files_as_snapshots)} of {len(midi_files)} files")

    return files_as_snapshots


def filter_piano_range(dataset_as_snapshots):
    filtered_dataset = []

    for filename, snapshots_track_1, snapshots_track_2 in dataset_as_snapshots:
        filtered_snapshots_track_1 = [snapshot[21:109] for snapshot in snapshots_track_1]
        filtered_snapshots_track_2 = [snapshot[21:109] for snapshot in snapshots_track_2]
        filtered_dataset.append((filename, np.array(filtered_snapshots_track_1), np.array(filtered_snapshots_track_2)))

    return filtered_dataset


def export_snapshots_to_csv(filtered_dataset, output_dir):
    for filename, snapshots_track_1, snapshots_track_2 in filtered_dataset:
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        csv_filename_track_1 = os.path.join(output_dir, f"{base_filename}_track_1.csv")
        csv_filename_track_2 = os.path.join(output_dir, f"{base_filename}_track_2.csv")

        if snapshots_track_1.size > 0:
            df_track_1 = pd.DataFrame(snapshots_track_1)
            df_track_1 = df_track_1.loc[:, (df_track_1 != 0).any(axis=0)]
            df_track_1.to_csv(csv_filename_track_1, index=False)

        if snapshots_track_2.size > 0:
            df_track_2 = pd.DataFrame(snapshots_track_2)
            df_track_2 = df_track_2.loc[:, (df_track_2 != 0).any(axis=0)]
            df_track_2.to_csv(csv_filename_track_2, index=False)

        print(f"Exported {csv_filename_track_1} and {csv_filename_track_2}")
