import pandas as pd
from mido import MidiFile, MidiTrack, Message

# Load the CSV files
predicted_harmony_df = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\predicted_melody\predicted_harmony.csv')
original_melody_df = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\predict_melody\AgeOfAquarius_melody.csv')

# Apply threshold to predicted harmony data
predicted_harmony_df = predicted_harmony_df.applymap(lambda x: 1 if x > 0.25 else 0)

# Create a new MIDI file and two tracks
mid = MidiFile()
melody_track = MidiTrack()
harmony_track = MidiTrack()
mid.tracks.append(melody_track)
mid.tracks.append(harmony_track)

# Constants
time_per_snapshot = 0.1  # seconds
ticks_per_beat = mid.ticks_per_beat
tempo = 500000  # microseconds per beat, equivalent to 120 BPM
ticks_per_snapshot = int(ticks_per_beat * (time_per_snapshot / (60 / 120)))  # for 120 BPM

# Initial states of the keys
previous_melody_keys = [0] * 88
previous_harmony_keys = [0] * 88

# Iterate over each row (snapshot) in the dataframes
for index in range(len(original_melody_df)):
    melody_keys = original_melody_df.iloc[index].tolist()
    harmony_keys = predicted_harmony_df.iloc[index].tolist()

    for key in range(88):
        # Handle melody track
        if melody_keys[key] == 1 and previous_melody_keys[key] == 0:
            # Note on
            melody_track.append(Message('note_on', note=key + 21, velocity=64, time=0))
        elif melody_keys[key] == 0 and previous_melody_keys[key] == 1:
            # Note off
            melody_track.append(Message('note_off', note=key + 21, velocity=64, time=0))

        # Handle harmony track
        if harmony_keys[key] == 1 and previous_harmony_keys[key] == 0:
            # Note on
            harmony_track.append(Message('note_on', note=key + 21, velocity=64, time=0))
        elif harmony_keys[key] == 0 and previous_harmony_keys[key] == 1:
            # Note off
            harmony_track.append(Message('note_off', note=key + 21, velocity=64, time=0))

    previous_melody_keys = melody_keys
    previous_harmony_keys = harmony_keys

    # Add time delay (advance time) for each track
    melody_track.append(Message('note_on', note=0, velocity=0, time=ticks_per_snapshot))
    harmony_track.append(Message('note_on', note=0, velocity=0, time=ticks_per_snapshot))

# Save the MIDI file
output_path = 'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\output_midi\lstm_output.mid'
mid.save(output_path)

print(f'MIDI file saved to {output_path}')