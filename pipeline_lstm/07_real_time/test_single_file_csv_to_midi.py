import os
import pandas as pd
from mido import MidiFile, MidiTrack, Message

from data_preperation.globals import INTERVAL

# Set a threshold value
threshold = 0.10

# Load the CSV file
csv_file = 'piano_keys_data.csv'
data_df = pd.read_csv(csv_file)

# Apply threshold to data
data_df = data_df.map(lambda x: 1 if x > threshold else 0)

# Create a new MIDI file and one track
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Constants
TIME_PER_SNAPSHOT = INTERVAL  # seconds
TICKS_PER_BEAT = mid.ticks_per_beat
TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (TIME_PER_SNAPSHOT / (60 / 120)))  # for 120 BPM

# Initial state of the keys
previous_keys = [0] * 88

# Iterate over each row (snapshot) in the dataframe
for index in range(len(data_df)):
    keys = data_df.iloc[index].tolist()

    for key in range(88):
        if keys[key] == 1 and previous_keys[key] == 0:
            # Note on
            track.append(Message('note_on', note=key + 21, velocity=64, time=0))
        elif keys[key] == 0 and previous_keys[key] == 1:
            # Note off
            track.append(Message('note_off', note=key + 21, velocity=64, time=0))

    previous_keys = keys

    # Add time delay (advance time) for the track
    track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

# Save the MIDI file
output_path = 'single_file_output/'
output_data_name = 'SF_output.mid'
if not os.path.exists(output_path):
    os.makedirs(output_path)

mid.save(f'{output_path}{output_data_name}')

print(f'MIDI file saved to {output_path}')