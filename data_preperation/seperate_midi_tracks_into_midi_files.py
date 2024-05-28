import os
import mido
from mido import MidiFile, MidiTrack


# Function to split MIDI tracks
def split_midi_tracks(input_file, output_dir):
    # Load the MIDI file
    mid = MidiFile(input_file)

    # Extract the original filename without extension
    original_filename = os.path.splitext(os.path.basename(input_file))[0]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the tracks and save each as a separate MIDI file
    for i, track in enumerate(mid.tracks):
        # Create a new MIDI file for each track
        new_midi = MidiFile()
        new_track = MidiTrack()

        # Copy the messages from the original track to the new track
        for msg in track:
            new_track.append(msg)

        # Append the new track to the new MIDI file
        new_midi.tracks.append(new_track)

        # Create the output filename
        output_file = os.path.join(output_dir, f'{original_filename}_track_{i + 1}.mid')

        # Save the new MIDI file
        new_midi.save(output_file)
        print(f'Saved {output_file}')


# Path to your input MIDI file
input_midi_file = '/home/falaxdb/Repos/minus1/datasets/temp/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav-split.midi'

# Specify the output directory
output_directory = '/home/falaxdb/Repos/minus1/datasets/temp/output'

# Split the MIDI tracks
split_midi_tracks(input_midi_file, output_directory)
