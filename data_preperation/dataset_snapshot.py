import mido
import os
import fnmatch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures


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
    based on their base patterns, ensuring each group has exactly one 'rightH' and one 'leftH' file.

    Args:
        root_dir (str): The root directory to start the search.
        pattern (str, optional): An optional pattern to filter the MIDI files. Only files
                                 containing this pattern in their names will be included.

    Returns:
        defaultdict: A dictionary where each key is a base pattern and the value is a dictionary
                     with 'rightH' and 'leftH' keys for corresponding file paths.

    Raises:
        ValueError: If any group does not have both 'rightH' and 'leftH' MIDI files.

    Example:
        midi_files = find_midi_files('/path/to/root_dir')
        for base_pattern, files in midi_files.items():
            print(f"Group: {base_pattern}")
            print(f" - Right Hand: {files['rightH']}")
            print(f" - Left Hand: {files['leftH']}")
    """
    midi_groups = defaultdict(dict)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename.lower(), '*.midi') or fnmatch.fnmatch(filename.lower(), '*.mid'):
                if pattern is None or fnmatch.fnmatch(filename.lower(), f'*{pattern.lower()}*'):
                    filepath = os.path.join(dirpath, filename)
                    parts = filename.lower().split('_')
                    if 'righth' in parts[-1]:
                        base_pattern = '_'.join(parts[:-1])
                        midi_groups[base_pattern]['rightH'] = filepath
                    elif 'lefth' in parts[-1]:
                        base_pattern = '_'.join(parts[:-1])
                        midi_groups[base_pattern]['leftH'] = filepath

    # Ensure each group has exactly one 'rightH' and one 'leftH'
    for base_pattern, files in midi_groups.items():
        if 'rightH' not in files or 'leftH' not in files:
            raise ValueError(f"Group {base_pattern} does not have both 'rightH' and 'leftH' MIDI files.")

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
    min_start = float('inf')
    max_end = 0

    for snapshots in group_snapshots:
        if len(snapshots) == 0:
            raise ValueError("Snapshot array was empty")

        non_empty_indices = np.where(snapshots.any(axis=1))[0]
        if non_empty_indices.size > 0:
            first_non_empty = non_empty_indices[0]
            last_non_empty = non_empty_indices[-1]
            if first_non_empty < min_start:
                min_start = first_non_empty
            if last_non_empty > max_end:
                max_end = last_non_empty

    trimmed_group_snapshots = [snapshots[min_start:max_end + 1] for snapshots in group_snapshots]

    return trimmed_group_snapshots


def __process_single_midi(midi_file, interval):
    """
    Helper function to process a single MIDI file and return its snapshots.

    Args:
        midi_file (str): The path to the MIDI file.
        interval (float): The interval (in seconds) at which snapshots are taken.

    Returns:
        tuple: A tuple containing the MIDI file path and the array of snapshots.
    """
    snapshots_array = snapshot_active_notes_from_midi(midi_file, interval)
    return midi_file, snapshots_array


def process_dataset(dataset_dir, interval, pattern=None, amount=0):
    """
    Processes a dataset of MIDI files, taking snapshots of active notes at specified intervals
    and grouping related files together.

    Args:
        dataset_dir (str): The directory containing the dataset of MIDI files.
        interval (float): The interval (in seconds) at which snapshots are taken.
        pattern (str, optional): An optional pattern to filter the MIDI files.
        amount (int, optional): An optional amount of how many songs should be processed.

    Returns:
        list: A list of snapshots for each group of MIDI files. The group will always have the right hand first [0]
        and then the left hand [1]
    """
    midi_files = find_midi_files(dataset_dir, pattern)

    # limit amount of files
    if amount > 0:
        midi_files = {k: midi_files[k] for k in list(midi_files)[:amount]}

    files_as_snapshots = []
    filenames = []

    total_files = sum(len(files) for files in midi_files.values())
    progress_bar = tqdm(total=total_files)

    for base_pattern, group_files in midi_files.items():
        group_snapshots = []
        for hand in ['rightH', 'leftH']:
            midi_file = group_files[hand]
            snapshots_array = snapshot_active_notes_from_midi(midi_file, interval)
            group_snapshots.append(snapshots_array)
            filenames.append(midi_file)
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

        trimmed_group_snapshots = trim_snapshots(group_snapshots)
        files_as_snapshots.append(trimmed_group_snapshots)

    progress_bar.close()

    return files_as_snapshots


def process_dataset_multithreaded(dataset_dir, interval, pattern=None, amount=0):
    """
    Processes a dataset of MIDI files, taking snapshots of active notes at specified intervals,
    grouping related files together, and using multithreading for efficiency.

    Args:
        dataset_dir (str): The directory containing the dataset of MIDI files.
        interval (float): The interval (in seconds) at which snapshots are taken.
        pattern (str, optional): An optional pattern to filter the MIDI files.
        amount (int, optional): An optional amount of how many songs should be processed.

    Returns:
        list: A list of snapshots for each group of MIDI files. The group will always have the right hand first [0]
        and then the left hand [1]
    """
    midi_files = find_midi_files(dataset_dir, pattern)

    # limit ammount of files
    if amount > 0:
        midi_files = {k: midi_files[k] for k in list(midi_files)[:amount]}

    files_as_snapshots = []

    total_files = sum(len(files) for files in midi_files.values())
    progress_bar = tqdm(total=total_files)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_midi = {executor.submit(__process_single_midi, midi_file, interval): midi_file
                          for group_files in midi_files.values() for midi_file in group_files.values()}

        for future in concurrent.futures.as_completed(future_to_midi):
            midi_file, snapshots_array = future.result()
            files_as_snapshots.append((midi_file, snapshots_array))
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

    progress_bar.close()

    grouped_snapshots = defaultdict(dict)
    for midi_file, snapshots_array in files_as_snapshots:
        base_pattern = '_'.join(os.path.basename(midi_file).lower().split('_')[:-1])
        if 'righth' in midi_file.lower():
            grouped_snapshots[base_pattern]['rightH'] = snapshots_array
        elif 'lefth' in midi_file.lower():
            grouped_snapshots[base_pattern]['leftH'] = snapshots_array

    final_grouped_snapshots = []
    for group in grouped_snapshots.values():
        if group['rightH'] is not None and group['leftH'] is not None:
            final_grouped_snapshots.append(trim_snapshots([group['rightH'], group['leftH']]))

    print(f"Processed {len(files_as_snapshots)} of {total_files} files")

    return final_grouped_snapshots


def snapshot_active_notes_from_midi_12keys(file_path, interval):
    """
    Processes a MIDI file and returns snapshots of active notes at specified intervals.

    Args:
        file_path (str): The path to the MIDI file.
        interval (float): The interval (in seconds) at which snapshots are taken.

    Returns:
        np.ndarray: An array of snapshots, where each snapshot is a list of 12 keys and a chord height.
    """
    mid = mido.MidiFile(file_path)
    snapshots = []
    active_notes = [0] * 12  # 12 keys
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

        if msg.type == 'note_on' or msg.type == 'note_off':
            key = msg.note % 12
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[key] = 1
            else:
                active_notes[key] = 0

        previous_event_time = current_time

    # Add the final snapshot if it hasn't been added yet
    snapshots.append(active_notes[:])

    return np.array(snapshots)


def process_dataset_12keys(dataset_dir, interval, pattern=None, amount=0):
    midi_files = find_midi_files(dataset_dir, pattern)

    # limit amount of files
    if amount > 0:
        midi_files = {k: midi_files[k] for k in list(midi_files)[:amount]}

    files_as_snapshots = []
    filenames = []

    total_files = sum(len(files) for files in midi_files.values())
    progress_bar = tqdm(total=total_files)

    for base_pattern, group_files in midi_files.items():
        group_snapshots = []
        for hand in ['rightH', 'leftH']:
            midi_file = group_files[hand]
            snapshots_array = snapshot_active_notes_from_midi_12keys(midi_file, interval)
            group_snapshots.append(snapshots_array)
            filenames.append(midi_file)
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

        trimmed_group_snapshots = trim_snapshots(group_snapshots)
        files_as_snapshots.append(trimmed_group_snapshots)

    progress_bar.close()

    return files_as_snapshots

    # Function to map MIDI note to octave position
def map_to_octave(note):
    return note % 12

# Function to compress a single track
def compress_track(track):
    compressed_track = np.zeros((track.shape[0], 12))
    for i, snapshot in enumerate(track):
        for note_index, is_active in enumerate(snapshot):
            if is_active:
                octave_position = map_to_octave(note_index)
                compressed_track[i][octave_position] = 1
    return compressed_track


def compress_existing_dataset_to_12keys(dataset):
    compressed_dataset = []
    for song in dataset:
        compressed_song = []
        for track in song:
            compressed_track = compress_track(track)
            compressed_song.append(compressed_track)
        compressed_dataset.append(compressed_song)
    return compressed_dataset


def filter_piano_range(grouped_snapshots):
    """
    Filters the snapshots to keep only the notes in the piano range (MIDI notes 21 to 108).

    Args:
        grouped_snapshots (list): A list of lists, where each sublist contains numpy arrays of snapshots
                                  for a group of MIDI files.

    Returns:
        list: A list of lists, where each sublist contains numpy arrays of filtered snapshots
              for a group of MIDI files, keeping only the piano range notes.
    """
    filtered_groups = []

    for group in grouped_snapshots:
        filtered_group = []
        for snapshots in group:
            filtered_snapshots = [snapshot[21:109] for snapshot in snapshots]
            filtered_group.append(np.array(filtered_snapshots))
        filtered_groups.append(filtered_group)

    return filtered_groups
