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
            # print(f"  Time step {time_step}, keys type: {type(keys)}, keys shape: {np.shape(keys)}")
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


def pad_to_88_keys(one_hot_vector, start_key=21, octaves_higher=0, total_keys=88):
    """
    Pad a one-hot encoded vector to fit 88 keys of a piano and place it a specified number of octaves higher.

    Parameters:
    one_hot_vector (np.ndarray): Input one-hot encoded vector.
    start_key (int): The starting key in the 88-key piano.
    octaves_higher (int): Number of octaves to shift the starting key higher.
    total_keys (int): The total number of keys on the piano (default is 88).

    Returns:
    np.ndarray: Padded one-hot encoded vector with 88 keys.
    """
    # Calculate the new starting key based on the number of octaves higher
    start_key = start_key + (octaves_higher * 12)

    # Initialize the full 88-key vector with zeros
    padded_vector = np.zeros(total_keys, dtype=int)
    end_key = start_key + len(one_hot_vector)

    if end_key > total_keys:
        raise ValueError("The one-hot vector is too long to fit in the 88 keys starting from the given start_key.")

    padded_vector[start_key:end_key] = one_hot_vector
    return padded_vector


def pad_sequence_of_one_hot_vectors(sequence, start_key=21, octaves_higher=0, total_keys=88):
    """
    Pad a sequence of one-hot encoded vectors to fit 88 keys of a piano, placing each one a specified number of octaves higher.

    Parameters:
    sequence (list of np.ndarray): Sequence of one-hot encoded vectors.
    start_key (int): The starting key in the 88-key piano.
    octaves_higher (int): Number of octaves to shift the starting key higher.
    total_keys (int): The total number of keys on the piano (default is 88).

    Returns:
    np.ndarray: 2D array where each row is a padded one-hot encoded vector with 88 keys.
    """
    padded_vectors = [pad_to_88_keys(vector, start_key, octaves_higher, total_keys) for vector in sequence]
    return np.stack(padded_vectors)