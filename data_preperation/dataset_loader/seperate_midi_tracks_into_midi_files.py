import os
import fnmatch
from mido import MidiFile, MidiTrack

def find_all_midi_files(root_dir):
    """
    Recursively searches for all MIDI files in the specified root directory and its subdirectories.

    Args:
        root_dir (str): The root directory to start the search.

    Returns:
        list: A list of paths to all found MIDI files.

    Example:
        midi_files = find_all_midi_files('/path/to/root_dir')
    """
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename.lower(), '*.midi') or fnmatch.fnmatch(filename.lower(), '*.mid'):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files

def split_midi_tracks(input_file, output_dir):
    """
    Splits the tracks of a MIDI file into separate MIDI files and saves them in the specified output directory.

    Args:
        input_file (str): The path to the input MIDI file.
        output_dir (str): The directory where the split MIDI files will be saved.

    Returns:
        None

    Example:
        split_midi_tracks('path/to/input_file.mid', 'path/to/output_dir')

    This function loads a MIDI file, iterates through its tracks, and saves each track as a separate MIDI file
    in the specified output directory. The output filenames are created by appending '_track_X' to the original
    filename, where X is the track number.
    """
    # Load the MIDI file
    mid = MidiFile(input_file)

    # Extract the original filename without extension
    original_filename = os.path.splitext(os.path.basename(input_file))[0]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the tracks and save each as a separate MIDI file
    for i, track in enumerate(mid.tracks):
        # Determine the track name based on the index
        track_name = 'rightH' if i == 0 else 'leftH' if i == 1 else f'surplus{i}'

        # Create a new MIDI file with a single track
        new_midi = MidiFile()
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        # Copy the messages from the original track to the new track
        for msg in track:
            new_track.append(msg)


        # Create the output filename
        output_file = os.path.join(output_dir, f'{original_filename}_{track_name}.mid')

        # Save the new MIDI file
        new_midi.save(output_file)
        print(f'Saved {output_file}')

def split_all_midi_files(root_dir, output_dir):
    """
    Finds all MIDI files in the specified root directory and its subdirectories, and splits each file into separate tracks.

    Args:
        root_dir (str): The root directory to start the search for MIDI files.
        output_dir (str): The directory where the split MIDI files will be saved.

    Returns:
        None

    Example:
        split_all_midi_files('/path/to/root_dir', '/path/to/output_dir')

    This function finds all MIDI files in the specified root directory and its subdirectories, splits each file
    into separate tracks, and saves the resulting files in the specified output directory.
    """
    midi_files = find_all_midi_files(root_dir)
    for midi_file in midi_files:
        split_midi_tracks(midi_file, output_dir)
