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
    # load midi-messages from file
    mid = mido.MidiFile(file_path)
    snapshots = []
    # no notes are played initially
    active_notes = [0] * 128
    current_time = 0
    snapshot_time = 0
    previous_event_time = 0

    # if iterating through the midi file, time is in seconds https://mido.readthedocs.io/en/stable/files/midi.html#about-the-time-attribute
    # if iterating through a track the time is in ticks
    # in this case the time is the delta time to the previous event in seconds
    for msg in mid:
        # Snapshot every intervall
        # all note on and of events exactly at intervall should be considered before taking snapshot
        # -> need to wait for all events on intervall before processing

        # Update time at which event takes place
        current_time += msg.time

        # Check if it's time to take a snapshot or if the elapsed time has exceeded the next snapshot time
        while current_time >= snapshot_time + interval:  # Take a snapshot every intervall
            # prevent snapshot to first process all snapshots happening at the same time
            if current_time == previous_event_time:
                break
            # Save the snapshot (copy the list to avoid reference issues)
            snapshots.append(active_notes[:])

            snapshot_time += interval

        # Process 'note_on' and 'note_off' events
        # note_on with a velocity of 0 count as note off events
        if msg.type == 'note_on':
            if msg.velocity == 0:
                active_notes[msg.note] = 0
            else:
                active_notes[msg.note] = 1

        elif msg.type == 'note_off':
            active_notes[msg.note] = 0

        previous_event_time = current_time

    snapshots = np.array(snapshots)

    # Remove initial and final empty snapshots
    start_idx = next((i for i, snapshot in enumerate(snapshots) if any(snapshot)), 0)
    end_idx = next((i for i, snapshot in enumerate(reversed(snapshots)) if any(snapshot)), len(snapshots))
    return snapshots[start_idx:len(snapshots) - end_idx]


def find_midi_files(root_dir, pattern=None):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename.lower(), '*.midi') or fnmatch.fnmatch(filename.lower(), '*.mid'):
                if pattern is None or fnmatch.fnmatch(filename.lower(), f'*{pattern.lower()}*'):
                    midi_files.append(os.path.join(dirpath, filename))
    return sorted(midi_files)


def align_snapshots(left_snapshots, right_snapshots):
    min_length = min(len(left_snapshots), len(right_snapshots))
    return left_snapshots[:min_length], right_snapshots[:min_length]


def process_dataset(dataset_dir, interval, use_all_data=True, amount_data=0, pattern=None):
    midi_files = find_midi_files(dataset_dir, pattern)

    # Limit amount of data if needed
    midi_files_length = len(midi_files) // 2
    if not use_all_data:
        midi_files = midi_files[:amount_data * 2]
        print("Using only", amount_data, "of", midi_files_length, "MIDI-files")
    else:
        print("Using all data of", midi_files_length, "MIDI-files")

    files_as_snapshots = []

    # Initialize tqdm with the number of MIDI files
    progress_bar = tqdm(total=len(midi_files) // 2)

    for i in range(0, len(midi_files), 2):
        left_file = midi_files[i]
        right_file = midi_files[i + 1]

        if "_leftH" in left_file and "_rightH" in right_file and left_file.replace("_leftH", "") == right_file.replace("_rightH", ""):
            left_snapshots = snapshot_active_notes_from_midi(left_file, interval)
            right_snapshots = snapshot_active_notes_from_midi(right_file, interval)
            left_snapshots, right_snapshots = align_snapshots(left_snapshots, right_snapshots)

            files_as_snapshots.append((left_file, left_snapshots))
            files_as_snapshots.append((right_file, right_snapshots))

            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")
        else:
            print(f"Unmatched pair: {left_file}, {right_file}")

    # Close the progress bar
    progress_bar.close()
    return files_as_snapshots


def __process_single_midi(midi_file, interval):
    snapshots_array = snapshot_active_notes_from_midi(midi_file, interval)
    return midi_file, snapshots_array


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


def extract_melody_and_harmony(dataset_as_snapshots):
    melody_harmony_dataset = []

    for filename, snapshots in dataset_as_snapshots:
        melody_snapshots = []
        harmony_snapshots = []
        for snapshot in snapshots:
            highest_note = max([note for note, active in enumerate(snapshot) if active], default=None)
            if highest_note is not None:
                melody_snapshot = [0] * 88
                harmony_snapshot = snapshot[:]
                melody_snapshot[highest_note] = 1
                harmony_snapshot[highest_note] = 0
                melody_snapshots.append(melody_snapshot)
                harmony_snapshots.append(harmony_snapshot)
            else:
                melody_snapshots.append([0] * 88)
                harmony_snapshots.append(snapshot[:])
        melody_harmony_dataset.append((filename, np.array(melody_snapshots), np.array(harmony_snapshots)))

    return melody_harmony_dataset


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

def filter_piano_range(dataset_as_snapshots):
    filtered_dataset = []

    for filename, snapshots in dataset_as_snapshots:
        filtered_snapshots = [snapshot[21:109] for snapshot in snapshots]
        filtered_dataset.append((filename, np.array(filtered_snapshots)))

    return filtered_dataset


def print_snapshot(snapshot):
    active_notes = [note_number for note_number, active in enumerate(snapshot) if active]
    if active_notes:
        print("Active notes:", ", ".join(map(str, active_notes)))
    else:
        print("No active notes.")


def print_snapshots(snapshots_array):
    for i, snapshot in enumerate(snapshots_array):
        print(f"Snapshot {i}:")
        print_snapshot(snapshot)
        print()


def print_dataset(dataset_as_snapshots):
    print("Number of files in Dataset:", len(dataset_as_snapshots))
    print("==============================")
    for filename, snapshots_array in dataset_as_snapshots:
        print(f"Snapshots of file {filename}:")
        print()
        print_snapshots(snapshots_array)
        print("==============================")


def print_melody_harmony_snapshots(melody_snapshots, harmony_snapshots):
    for i, (melody_snapshot, harmony_snapshot) in enumerate(zip(melody_snapshots, harmony_snapshots)):
        print(f"Snapshot {i}:")
        print("Melody:")
        print_snapshot(melody_snapshot)
        print("Harmony:")
        print_snapshot(harmony_snapshot)
        print()


def print_melody_harmony_dataset(dataset_as_snapshots):
    print("Number of files in Dataset:", len(dataset_as_snapshots))
    print("==============================")
    for filename, melody_snapshots, harmony_snapshots in dataset_as_snapshots:
        print(f"File: {filename}")
        print("Melody and Harmony Snapshots:")
        print_melody_harmony_snapshots(melody_snapshots, harmony_snapshots)
        print("==============================")
