import os
from mido import MidiFile, MidiTrack


def split_midi_tracks(input_folder, output_folder, use_all_data=True, amount_data=0):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_count = 0
    # Iterate through each MIDI file in the input folder
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
                    print(f"Saved: {output_midi_path}")
                except IOError as e:
                    print(f"Error: Could not save {output_midi_path}. {e}")
