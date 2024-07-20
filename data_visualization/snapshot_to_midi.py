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
    """

    # Check if mido library is available
    try:
        import mido
    except ImportError as e:
        raise ImportError(
            "The mido library is required to run this function. Please install it using 'pip install mido'.") from e

    # Validate input parameters
    if not isinstance(snapshots, list) or not all(isinstance(snap, np.ndarray) for snap in snapshots):
        raise ValueError("snapshots must be a list of numpy.ndarray.")
    if not isinstance(track_names, list) or not all(isinstance(name, str) for name in track_names):
        raise ValueError("track_names must be a list of strings.")
    if not isinstance(time_per_snapshot, (int, float)) or time_per_snapshot <= 0:
        raise ValueError("time_per_snapshot must be a positive number.")
    if not isinstance(output_path, str) or not os.path.isdir(output_path):
        raise ValueError("output_path must be a valid directory path.")
    if not isinstance(output_name, str) or not output_name.endswith('.mid'):
        raise ValueError("output_name must be a string ending with '.mid'.")

    # Check if the number of snapshots matches the number of track names
    if len(snapshots) != len(track_names):
        raise ValueError("The length of snapshots and track_names must be the same.")

    # Create a new MIDI file
    midi_file = mido.MidiFile()

    # Process each track
    for i, (snapshot, track_name) in enumerate(zip(snapshots, track_names)):
        if not (isinstance(snapshot, np.ndarray) and snapshot.ndim == 2 and snapshot.shape[1] == 88):
            raise ValueError(f"Each snapshot must be a 2D numpy array with 88 columns (keys). Error in snapshot {i}.")

        track = mido.MidiTrack()
        track.name = track_name
        midi_file.tracks.append(track)

        # Initialize previous snapshot to handle note off events
        prev_snapshot = np.zeros(88, dtype=int)

        for time_step in snapshot:
            time_step = np.asarray(time_step)
            if time_step.shape[0] != 88:
                raise ValueError(f"Each row in snapshot {i} must have 88 columns. Found {time_step.shape[0]} columns.")

            # Calculate the elapsed time for each snapshot
            time_in_ticks = mido.second2tick(time_per_snapshot, ticks_per_beat=midi_file.ticks_per_beat,
                                             tempo=mido.bpm2tempo(120))

            # Note on events
            for key in range(88):
                if time_step[key] == 1 and prev_snapshot[key] == 0:
                    note_on = mido.Message('note_on', note=key + 21, velocity=64, time=0)
                    track.append(note_on)
                elif time_step[key] == 0 and prev_snapshot[key] == 1:
                    note_off = mido.Message('note_off', note=key + 21, velocity=64, time=0)
                    track.append(note_off)

            # Update previous snapshot
            prev_snapshot = time_step.copy()

            # Advance time for the next snapshot
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=int(time_in_ticks)))

        # Add a note off event for any remaining active notes at the end of the track
        for key in range(88):
            if prev_snapshot[key] == 1:
                note_off = mido.Message('note_off', note=key + 21, velocity=64, time=0)
                track.append(note_off)

    # Save the MIDI file
    output_file_path = os.path.join(output_path, output_name)
    try:
        midi_file.save(output_file_path)
        print(f"MIDI file saved to {output_file_path}")
    except Exception as e:
        raise IOError(f"Could not save MIDI file to {output_file_path}.") from e


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


def pad_beginning_of_sequence(sequence, amount_of_padding_snapshots):
    if len(sequence.shape) != 2:
        raise ValueError(f"The input sequence must be a 2D array. Provided array was of shape: {sequence.shape}")

    padding_sequence = np.full((amount_of_padding_snapshots, sequence.shape[1]), 0)

    return np.concatenate((padding_sequence, sequence), axis=0)


def split_snapshots_in_sequence(sequence):
    if len(sequence.shape) != 2:
        raise ValueError(f"The input sequence must be a 2D array. Provided array was of shape: {sequence.shape}")

    # print("Sequence shape:", sequence.shape)
    vec_length = sequence.shape[1]  # Should be size(1) for the correct dimension
    midpoint = vec_length // 2

    harmony = sequence[:, :midpoint]
    melody = sequence[:, midpoint:]

    return melody,harmony


def split_and_pad_sequence(sequence, amount_of_padding_snapshots):
    # Todo: improve by padding first, then splitting
    melody, harmony = split_snapshots_in_sequence(sequence)

    melody = pad_beginning_of_sequence(melody, amount_of_padding_snapshots)
    harmony = pad_beginning_of_sequence(harmony, amount_of_padding_snapshots)

    return melody, harmony
