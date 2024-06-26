import os

import pandas as pd
from mido import MidiFile, MidiTrack, Message

from data_preperation.globals import INTERVAL

"""
This script converts the predicted harmony and the original melody into a MIDI-file.
The value 'threshold' determines at what threshold the one-hot-encoding probability actually plays the key or doesnt play it.
If the threshold is high, it only plays notes the model is really sure about.
If the threshold is low, it plays more notes, even tho the model is not really sure, if they fit.
This version uses the 12-key data model.
"""
threshold = 0.25


# Load CSV files of predicted MIDI
predicted_harmony_df = pd.read_csv(
    '../05_inference/predicted_leftH/predicted_harmony.csv')
original_melody_df = pd.read_csv('../05_inference/predicted_leftH/original_melody.csv')


"""
# Load CSV files of realtime pitch
predicted_harmony_df = pd.read_csv('../07_real_time/predicted_data.csv')
original_melody_df = pd.read_csv('../07_real_time/pitch_data.csv')
# Apply threshold to predicted harmony data
"""
# TODO: Auslagern in function, damit ich das nicht immer umstellen muss...
"""
# Load CSV files of single_file tests
predicted_harmony_df = pd.read_csv('../420_developer_tests_only/single_file_output/predicted_harmony.csv')
original_melody_df = pd.read_csv('../420_developer_tests_only/single_file_output/song_1_rightH.csv')
"""
# Apply threshold to predicted harmony data
predicted_harmony_df = predicted_harmony_df.map(lambda x: 1 if x > threshold else 0)


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
previous_melody_keys = [0] * 12
previous_harmony_keys = [0] * 12

# Iterate over each row (snapshot) in the dataframes
for index in range(len(original_melody_df)):
    melody_keys = original_melody_df.iloc[index].tolist()
    harmony_keys = predicted_harmony_df.iloc[index].tolist()

    # +48 for 4 octaves higher, +36 for 3 octaves higher
    for key in range(12):
        # Handle melody track
        if melody_keys[key] == 1 and previous_melody_keys[key] == 0:
            # Note on
            melody_track.append(Message('note_on', note=key + 21 + 48, velocity=64, time=0))
        elif melody_keys[key] == 0 and previous_melody_keys[key] == 1:
            # Note off
            melody_track.append(Message('note_off', note=key + 21 + 48, velocity=64, time=0))

        # Handle harmony track
        if harmony_keys[key] == 1 and previous_harmony_keys[key] == 0:
            # Note on
            harmony_track.append(Message('note_on', note=key + 21 + 36, velocity=64, time=0))
        elif harmony_keys[key] == 0 and previous_harmony_keys[key] == 1:
            # Note off
            harmony_track.append(Message('note_off', note=key + 21 + 36, velocity=64, time=0))

    previous_melody_keys = melody_keys
    previous_harmony_keys = harmony_keys

    # Add time delay (advance time) for each track
    melody_track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))
    harmony_track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

# Save the MIDI file
output_path = 'output_mid/'
output_data_name = 'output.mid'
if not os.path.exists(output_path):
    os.makedirs(output_path)

mid.save(f'{output_path}{output_data_name}')

print(f'MIDI file saved to {output_path}')
