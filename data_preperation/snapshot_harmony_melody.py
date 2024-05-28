import mido

import os
import fnmatch
from collections import defaultdict

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
    """
    Processes a MIDI file and returns snapshots of active notes at specified intervals.

    Args:
        file_path (str): The path to the MIDI file.
        interval (float): The interval (in seconds) at which snapshots are taken.

    Returns:
        np.ndarray: An array of snapshots, where each snapshot is a list of active notes.
    """
    mid = mido.MidiFile(file_path)
    snapshots = []
    active_notes = [0] * 128
    current_time = 0
    snapshot_time = 0
    previous_event_time = 0

    for msg in mid:
        current_time += msg.time

        while current_time >= snapshot_time + interval:
            if current_time == previous_event_time:
                break
            snapshots.append(active_notes[:])
            snapshot_time += interval

        if msg.type == 'note_on':
            if msg.velocity == 0:
                active_notes[msg.note] = 0
            else:
                active_notes[msg.note] = 1
        elif msg.type == 'note_off':
            active_notes[msg.note] = 0

        previous_event_time = current_time

    return np.array(snapshots)

def find_midi_files(root_dir, pattern=None):
    """
    Recursively searches for MIDI files in the specified root directory and groups them
    based on their base patterns, excluding the track and track number suffix.

    Args:
        root_dir (str): The root directory to start the search.
        pattern (str, optional): An optional pattern to filter the MIDI files. Only files
                                 containing this pattern in their names will be included.

    Returns:
        defaultdict: A dictionary where each key is a base pattern and the value is a list
                     of file paths that match that base pattern.

    Example:
        midi_files = find_midi_files('/path/to/root_dir')
        for base_pattern, files in midi_files.items():
            print(f"Group: {base_pattern}")
            for file in files:
                print(f" - {file}")
    """
    midi_groups = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename.lower(), '*.midi') or fnmatch.fnmatch(filename.lower(), '*.mid'):
                if pattern is None or fnmatch.fnmatch(filename.lower(), f'*{pattern.lower()}*'):
                    filepath = os.path.join(dirpath, filename)
                    base_pattern = '_'.join(filename.lower().split('_')[:-2])
                    midi_groups[base_pattern].append(filepath)

    return midi_groups

def trim_snapshots(group_snapshots):
    """
    Trims the leading and trailing empty snapshots for each group of snapshots.

    Args:
        group_snapshots (list): A list of numpy arrays where each array represents snapshots
                                for a MIDI file in the group.

    Returns:
        list: A list of trimmed numpy arrays with empty snapshots removed from the beginning and end.
    """
    # Determine the minimum starting index with a non-empty snapshot
    min_start = float('inf')
    max_end = 0

    for snapshots in group_snapshots:
        # Find the first non-empty snapshot index
        non_empty_indices = np.where(snapshots.any(axis=1))[0]
        if non_empty_indices.size > 0:
            first_non_empty = non_empty_indices[0]
            last_non_empty = non_empty_indices[-1]
            if first_non_empty < min_start:
                min_start = first_non_empty
            if last_non_empty > max_end:
                max_end = last_non_empty

    # Trim the snapshots for each file in the group
    trimmed_group_snapshots = [snapshots[min_start:max_end+1] for snapshots in group_snapshots]

    return trimmed_group_snapshots

def process_dataset(dataset_dir, interval, pattern=None):
    """
    Processes a dataset of MIDI files, taking snapshots of active notes at specified intervals
    and grouping related files together.

    Args:
        dataset_dir (str): The directory containing the dataset of MIDI files.
        interval (float): The interval (in seconds) at which snapshots are taken.
        pattern (str, optional): An optional pattern to filter the MIDI files.

    Returns:
        list: A list of snapshots for each group of MIDI files.
    """
    midi_files = find_midi_files(dataset_dir, pattern)

    files_as_snapshots = []
    filenames = []

    # Initialize tqdm with the number of MIDI groups
    total_files = sum(len(files) for files in midi_files.values())
    progress_bar = tqdm(total=total_files)

    for base_pattern, group_files in midi_files.items():
        group_snapshots = []
        for midi_file in group_files:
            snapshots_array = snapshot_active_notes_from_midi(midi_file, interval)
            group_snapshots.append(snapshots_array)
            filenames.append(midi_file)
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

        # Trim the snapshots to remove leading and trailing empty snapshots
        trimmed_group_snapshots = trim_snapshots(group_snapshots)
        files_as_snapshots.append(trimmed_group_snapshots)

    progress_bar.close()

    return files_as_snapshots
