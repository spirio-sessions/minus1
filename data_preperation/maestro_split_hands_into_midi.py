import os
from mido import MidiFile, MidiTrack
from tqdm import tqdm


def split_midi_tracks(input_folder, output_folder, use_all_data=True, amount_data=0):
    """
    Split MIDI files into separate tracks and save them to an output folder.

    This function reads MIDI files from the specified input folder, splits each
    file into individual tracks, and saves each track as a new MIDI file in the
    output folder. The tracks are named based on their index: the first track is
    named 'rightH', and subsequent tracks are named 'leftH'. The function can also
    limit the number of files processed if specified.

    Parameters:
    input_folder (str): The path to the folder containing the input MIDI files.
    output_folder (str): The path to the folder where the split MIDI tracks will be saved.
    use_all_data (bool, optional): If True, all files in the input folder will be processed.
                                   If False, only 'amount_data' files will be processed. Default is True.
    amount_data (int, optional): The number of files to process if 'use_all_data' is False. Default is 0.

    Returns:
    None

    Example:
    >>> input_folder = './midi_files'
    >>> output_folder = './split_midi_tracks'
    >>> split_midi_tracks(input_folder, output_folder, use_all_data=False, amount_data=10)
    Limiting data to 10 files.
    Processed dataset (10/10)
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_count = 0
    # Iterate through each MIDI file in the input folder
    progress_bar = tqdm(total=len(os.listdir(input_folder)))
    for filename in os.listdir(input_folder):
        if filename.endswith(".midi"):
            # Limits the data if needed
            if not use_all_data and data_count == 0:
                print("Limiting data to", amount_data, "files.")
            if not use_all_data and data_count == amount_data:
                break
            else:
                data_count += 1
            input_midi_path = os.path.join(input_folder, filename)
            try:
                midi = MidiFile(input_midi_path)
            except EOFError:
                print(f"Error: Could not read {input_midi_path}. Skipping file.")
                continue

            for i, track in enumerate(midi.tracks):
                # Determine the track name based on the index
                track_name = 'rightH' if i == 0 else 'leftH'

                # Create a new MIDI file with a single track
                new_midi = MidiFile()
                new_track = MidiTrack()
                new_midi.tracks.append(new_track)

                # Copy the messages from the original track to the new track
                for msg in track:
                    new_track.append(msg)

                # Construct the output file path
                output_filename = f"{os.path.splitext(filename)[0]}_{track_name}.mid"
                output_midi_path = os.path.join(output_folder, output_filename)

                # Save the new MIDI file
                try:
                    # Save the new MIDI file
                    new_midi.save(output_midi_path)
                except IOError as e:
                    print(f"Error: Could not save {output_midi_path}. {e}")

        progress_bar.update(1)
        progress_bar.set_description(f"Processed dataset ({progress_bar.n}/{progress_bar.total})")
