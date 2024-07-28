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

# Load CSV files of predicted MIDI
original_melody_df = pd.read_csv('../05_inference/predicted_leftH/original_melody.csv')
original_harmony_df = pd.read_csv('../05_inference/predicted_leftH/original_harmony.csv')

predicted_data_df = pd.read_csv('../05_inference/predicted_leftH/predicted_data.csv')
predicted_harmony_df = pd.read_csv('../05_inference/predicted_leftH/predicted_harmony.csv')

# Apply threshold to predicted harmony data
threshold = 0.15
predicted_harmony_df = predicted_harmony_df.map(lambda x: 1 if x > threshold else 0)
predicted_data = predicted_harmony_df.map(lambda x: 1 if x > threshold else 0)

tracks = [original_melody_df, original_harmony_df, predicted_data_df, predicted_harmony_df]
track_names = ["Original Melody", "Original Harmony", "Generated Data", "Predicted Harmony"]
octaves_higher = [48, 36, 36, 36]

# Create a new MIDI file and two tracks
mid = MidiFile()

# Constants
TIME_PER_SNAPSHOT = INTERVAL  # seconds
TICKS_PER_BEAT = mid.ticks_per_beat
TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (TIME_PER_SNAPSHOT / (60 / 120)))  # for 120 BPM

for i, (data, track_name, octave_higher) in enumerate(zip(tracks, track_names, octaves_higher)):
    track = MidiTrack()
    track.name = track_name
    mid.tracks.append(track)

    previous_keys = [0] * 24
    for index in range(len(data)):
        track_keys = data.iloc[index].tolist()
        for key in range(data.shape[1]):
            if track_keys[key] == 1 and previous_keys[key] == 0:
                # Note on
                track.append(Message('note_on', note=key + 21 + octave_higher, velocity=64, time=0))
            elif track_keys[key] == 0 and previous_keys[key] == 1:
                # Note off
                track.append(Message('note_off', note=key + 21 + octave_higher, velocity=64, time=0))
        previous_keys = track_keys.copy()
        track_keys.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

# Save the MIDI file
output_path = 'output_mid/'
output_data_name = 'output.mid'
if not os.path.exists(output_path):
    os.makedirs(output_path)

mid.save(f'{output_path}{output_data_name}')

print(f'MIDI file saved to {output_path}')
