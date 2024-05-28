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

    if len(mid.tracks) < 2:
        raise ValueError("MIDI file must contain at least two tracks.")

    # Initialize variables for track 0 and track 1
    snapshots_track_0 = []
    snapshots_track_1 = []
    active_notes_track_0 = [0] * 128
    active_notes_track_1 = [0] * 128
    current_time_0 = 0
    current_time_1 = 0
    snapshot_time_0 = 0
    snapshot_time_1 = 0
    previous_event_time_0 = 0
    previous_event_time_1 = 0

    # Process track 0
    for msg in mid.tracks[0]:
        current_time_0 += msg.time

        while current_time_0 >= snapshot_time_0 + interval:
            # prevent snapshot to first process all snapshots happening at the same time
            if current_time_0 == previous_event_time_0:
                break

            snapshots_track_0.append(active_notes_track_0[:])
            snapshot_time_0 += interval

        if msg.type == 'note_on' or msg.type == 'note_off':
            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                active_notes_track_0[msg.note] = 0
            else:
                active_notes_track_0[msg.note] = 1

        previous_event_time_0 = current_time_0

    # Convert to numpy arrays
    snapshots_track_0 = np.array(snapshots_track_0)

    # Remove initial and final empty snapshots
    start_idx_track_0 = next((i for i, snapshot in enumerate(snapshots_track_0) if any(snapshot)), 0)
    end_idx_track_0 = next((i for i, snapshot in enumerate(reversed(snapshots_track_0)) if any(snapshot)), len(snapshots_track_0))

    # Process track 1
    for msg in mid.tracks[1]:
        current_time_1 += msg.time

        while current_time_1 >= snapshot_time_1 + interval:
            # prevent snapshot to first process all snapshots happening at the same time
            if current_time_1 == previous_event_time_1:
                break

            snapshots_track_1.append(active_notes_track_1[:])
            snapshot_time_1 += interval

        if msg.type == 'note_on' or msg.type == 'note_off':
            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                active_notes_track_1[msg.note] = 0
            else:
                active_notes_track_1[msg.note] = 1

        previous_event_time_1 = current_time_1

    # Convert to numpy arrays
    snapshots_track_1 = np.array(snapshots_track_1)

    # Remove initial and final empty snapshots
    start_idx_track_1 = next((i for i, snapshot in enumerate(snapshots_track_1) if any(snapshot)), 0)
    end_idx_track_1 = next((i for i, snapshot in enumerate(reversed(snapshots_track_1)) if any(snapshot)), len(snapshots_track_1))

    return snapshots_track_0[start_idx_track_0:len(snapshots_track_0) - end_idx_track_0], snapshots_track_1[start_idx_track_1:len(snapshots_track_1) - end_idx_track_1]


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

    for filename, snapshots_track_0, snapshots_track_1 in dataset_as_snapshots:
        filtered_snapshots_track_0 = [snapshot[21:109] for snapshot in snapshots_track_0]
        filtered_snapshots_track_1 = [snapshot[21:109] for snapshot in snapshots_track_1]
        filtered_dataset.append((filename, np.array(filtered_snapshots_track_0), np.array(filtered_snapshots_track_1)))

    return filtered_dataset


def export_snapshots_to_csv(filtered_dataset, output_dir):
    for filename, snapshots_track_0, snapshots_track_1 in filtered_dataset:
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        csv_filename_track_0 = os.path.join(output_dir, f"{base_filename}_rightH.csv")
        csv_filename_track_1 = os.path.join(output_dir, f"{base_filename}_leftH.csv")

        if snapshots_track_0.size > 0:
            df_track_0 = pd.DataFrame(snapshots_track_0)
            df_track_0 = df_track_0.loc[:, (df_track_0 != 0).any(axis=0)]
            df_track_0.to_csv(csv_filename_track_0, index=False)

        if snapshots_track_1.size > 0:
            df_track_1 = pd.DataFrame(snapshots_track_1)
            df_track_1 = df_track_1.loc[:, (df_track_1 != 0).any(axis=0)]
            df_track_1.to_csv(csv_filename_track_1, index=False)

        print(f"Exported {csv_filename_track_0} and {csv_filename_track_1}")
