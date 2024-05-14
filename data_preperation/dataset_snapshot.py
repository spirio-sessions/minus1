import mido

import os
import fnmatch

# for progress bar:
from tqdm import tqdm

#for multithread:
import concurrent.futures


# interval in seconds
# function will process all events happening in an intervall
# including intervall start, excluding intervall end
# snapshot will show all active notes at the end of the intervall
def snapshot_active_notes_from_midi(file_path, interval):
    # load midi-messages from file
    mid = mido.MidiFile(file_path)

    snapshots = []
    snapshot_times = []
    # no notes are played initially
    active_notes = [False] * 128

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
        #

        # Update time at which event takes place
        current_time += msg.time

        # Check if it's time to take a snapshot or if the elapsed time has exceeded the next snapshot time
        while current_time >= snapshot_time + interval:  # Take a snapshot every intervall

            # prevent snapshot to first process all snapshots happening at the same time
            if current_time == previous_event_time:
                break

            # Save the snapshot (copy the list to avoid reference issues)
            snapshots.append(active_notes[:])
            snapshot_times.append(snapshot_time)

            snapshot_time += interval

        # Process 'note_on' and 'note_off' events
        # note_on with a velocity of 0 count as note off events
        if msg.type == 'note_on':
            if msg.velocity == 0:
                active_notes[msg.note] = False
            else:
                active_notes[msg.note] = True

        elif msg.type == 'note_off':
            active_notes[msg.note] = False

        previous_event_time = current_time

    snapshots_with_times = list(zip(snapshots, snapshot_times))

    # Truncate snapshots at the beginning and end where no notes are active
    # might be unneccesary?
    start_idx = 0
    for snapshot, _ in snapshots_with_times:
        if any(snapshot):
            break
        start_idx += 1

    end_idx = len(snapshots_with_times)
    for snapshot, _ in reversed(snapshots_with_times):
        if any(snapshot):
            break
        end_idx -= 1

    return snapshots_with_times[start_idx:end_idx]


def find_midi_files(root_dir):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.midi') or fnmatch.fnmatch(filename, '*.mid'):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files


def process_dataset(dataset_dir, interval):
    midi_files = find_midi_files(dataset_dir)

    files_as_snapshots = []

    # Initialize tqdm with the number of MIDI files
    progress_bar = tqdm(total=len(midi_files))

    for midi_file in midi_files:
        snapshots_with_times = snapshot_active_notes_from_midi(midi_file, interval)
        files_as_snapshots.append((midi_file, snapshots_with_times))

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

    # Close the progress bar
    progress_bar.close()

    return files_as_snapshots


def __process_single_midi(midi_file, interval):
    snapshots_with_times = snapshot_active_notes_from_midi(midi_file, interval)
    return (midi_file, snapshots_with_times)


def process_dataset_multithreaded(dataset_dir, interval):
    midi_files = find_midi_files(dataset_dir)

    files_as_snapshots = []

    # Split MIDI files into chunks for processing
    num_chunks = min(len(midi_files), os.cpu_count() or 1)  # Use CPU count as a default if os.cpu_count() returns None
    midi_file_chunks = [midi_files[i::num_chunks] for i in range(num_chunks)]

    # Initialize tqdm with the total number of MIDI files
    progress_bar = tqdm(total=len(midi_files))

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each chunk of MIDI files for processing
        futures = []
        for midi_files_chunk in midi_file_chunks:
            futures.extend(executor.submit(__process_single_midi, midi_file, interval) for midi_file in midi_files_chunk)

        # Iterate through the completed futures to retrieve the results
        for future in concurrent.futures.as_completed(futures):
            files_as_snapshots.append(future.result())
            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")

    # Close the progress bar
    progress_bar.close()

    print(f"processed {len(files_as_snapshots)} of {len(midi_files)} files")

    return files_as_snapshots


def print_snapshot(snapshot):
    active_notes = [note_number for note_number, active in enumerate(snapshot) if active]
    if active_notes:
        print("Active notes:", ", ".join(map(str, active_notes)))
    else:
        print("No active notes.")


def print_snapshots(snapshots_with_times):
    for i, (snapshot, time) in enumerate(snapshots_with_times):
        print(f"Snapshot {i} at time {time:.2f}s:")
        print_snapshot(snapshot)
        print()


def print_dataset(dataset_as_snapshots):
    print("Number of files in Dataset:", len(dataset_as_snapshots))
    print("==============================")

    for filename, snapshots in dataset_as_snapshots:
        print(f"Snapshots of file {filename}:")
        print()
        print_snapshots(snapshots)
        print("==============================")