import pandas as pd
from mido import MidiFile, MidiTrack, Message

from data_preperation.globals import INTERVAL


# Load the CSV files
predicted_harmony_df = pd.read_csv(
    '../../datasets/maestro_v3_split/small_batch_lstm/predicted_leftH/predicted_harmony.csv')
original_melody_df = pd.read_csv('../../datasets/maestro_v3_split/small_batch_lstm/predicted_leftH/original_melody.csv')

# Apply threshold to predicted harmony data
predicted_harmony_df = predicted_harmony_df.map(lambda x: 1 if x > 0.15 else 0)

# Create a new MIDI file and two tracks
mid = MidiFile()
melody_track = MidiTrack()
harmony_track = MidiTrack()
mid.tracks.append(melody_track)
mid.tracks.append(harmony_track)

# Constants
TIME_PER_SNAPSHOT = INTERVAL  # seconds
TICKS_PER_BEAT = mid.ticks_per_beat
TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (TIME_PER_SNAPSHOT / (60 / 120)))  # for 120 BPM

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
    melody_track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))
    harmony_track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

# Save the MIDI file
output_path = 'output_midi/output.mid'
mid.save(output_path)

print(f'MIDI file saved to {output_path}')
