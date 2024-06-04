from data_preperation import dataset_snapshot

dataset_dir = "/home/falaxdb/Repos/minus1/datasets/maestro_v3_split/hands_split_into_seperate_midis"

data = dataset_snapshot.process_dataset(dataset_dir, 0.1, amount=4)

filtered_data = dataset_snapshot.filter_piano_range(data)

for song in filtered_data:
    print("song:")
    for track in song:
        print(track.shape)

import mido
from mido import MidiFile, MidiTrack, Message
import os


def snapshots_to_midi(filtered_data, output_directory='.', output_filename='output.mid'):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the full path for the output file
    output_file = os.path.join(output_directory, output_filename)

    # Create a new MIDI file
    mid = MidiFile()

    for song in filtered_data:
        for track_data in song:
            track = MidiTrack()
            mid.tracks.append(track)

            # Store the state of the notes
            current_notes = [0] * 88

            # Iterate through each snapshot
            for snapshot_index, snapshot in enumerate(track_data):
                # Convert snapshot index to time (assuming 0.1 second per snapshot and 480 ticks per beat)
                time = int(0.1 * 480 / 0.5)  # 0.5 seconds per beat for a tempo of 120 bpm

                for note_index, is_on in enumerate(snapshot):
                    note_number = note_index + 21  # Mapping to MIDI note numbers

                    if is_on and not current_notes[note_index]:
                        # Note on event if the note was previously off
                        track.append(Message('note_on', note=note_number, velocity=64, time=0))
                        current_notes[note_index] = 1
                    elif not is_on and current_notes[note_index]:
                        # Note off event if the note was previously on
                        track.append(Message('note_off', note=note_number, velocity=64, time=0))
                        current_notes[note_index] = 0

                # Add a time delay to the next snapshot
                track.append(Message('note_off', note=0, velocity=0, time=time))

    # Save the MIDI file
    mid.save(output_file)



output_directory = '/home/falaxdb/Repos/minus1/datasets/temp/output'
snapshots_to_midi(filtered_data, output_directory, 'output.mid')
