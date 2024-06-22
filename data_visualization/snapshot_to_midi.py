import os
from mido import MidiFile, MidiTrack, Message
import numpy as np


def create_midi_from_snapshots(snapshots, track_names, time_per_snapshot, output_path, output_name):
    """
    Generate a MIDI file from a list of snapshot numpy arrays representing musical notes.

    Each snapshot in the list should be a 2D numpy array where each row corresponds to a track,
    and each column corresponds to a key (88 keys for a standard piano). A value of 1 indicates
    that the key is pressed, and 0 indicates that the key is not pressed.

    Parameters:
    -----------
    snapshots : list of numpy.ndarray
        List of 2D numpy arrays representing snapshots of key states. Each array should have
        dimensions (number of time steps, 88). The length of each snapshot array can vary.
    track_names : list of str
        List of track names to be used in the MIDI file. The length of this list should match
        the number of snapshots provided.
    time_per_snapshot : float
        Time duration of each snapshot in seconds.
    output_path : str
        Path to the directory where the MIDI file will be saved.
    output_name : str
        Name of the output MIDI file.

    Returns:
    --------
    None

    Example:
    --------
    >>> snapshots = [np.array([[0, 1, 0, ..., 0], [1, 0, 0, ..., 0]]), np.array([[0, 0, 1, ..., 0], [0, 1, 0, ..., 0]])]
    >>> track_names = ['melody', 'harmony']
    >>> time_per_snapshot = 0.1
    >>> output_path = '/path/to/output/'
    >>> output_name = 'output_name.mid'
    >>> create_midi_from_snapshots(snapshots, track_names, time_per_snapshot, output_path, output_name)
    MIDI file saved to /path/to/output/
    """
    # Create a new MIDI file
    mid = MidiFile()

    # Create and add tracks to the MIDI file
    tracks = []
    for track_name in track_names:
        track = MidiTrack()
        mid.tracks.append(track)
        tracks.append(track)

    # Constants
    TICKS_PER_BEAT = mid.ticks_per_beat
    TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
    TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (time_per_snapshot / (60 / 120)))  # for 120 BPM

    # Process each track independently
    for track_index, snapshot in enumerate(snapshots):
        track = tracks[track_index]
        previous_keys = [0] * 88

        print(f"Processing track {track_index}: {track_names[track_index]} with snapshot shape {snapshot.shape}")

        for time_step, keys in enumerate(snapshot):
            print(f"  Time step {time_step}, keys type: {type(keys)}, keys shape: {np.shape(keys)}")
            if not isinstance(keys, (list, tuple, np.ndarray)):
                print(f"Unexpected type for keys at time_step {time_step}, track_index {track_index}: {type(keys)}")
            for key in range(88):
                if keys[key] == 1 and previous_keys[key] == 0:
                    # Note on
                    track.append(Message('note_on', note=key + 21, velocity=64, time=0))
                elif keys[key] == 0 and previous_keys[key] == 1:
                    # Note off
                    track.append(Message('note_off', note=key + 21, velocity=64, time=0))
            previous_keys = keys

            # Add time delay (advance time)
            track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

    # Save the MIDI file
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mid.save(os.path.join(output_path, output_name))

    print(f'MIDI file saved to {output_path}')