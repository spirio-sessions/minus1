import os

import pandas as pd
from data_preperation.globals import INTERVAL
from mido import Message, MidiTrack, MidiFile

# Initiate Midifile
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Constants
TIME_PER_SNAPSHOT = INTERVAL  # seconds
TICKS_PER_BEAT = mid.ticks_per_beat
TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (TIME_PER_SNAPSHOT / (60 / 120)))  # for 120 BPM

# Initial states of the keys
previous_melody_keys = [0] * 88
previous_harmony_keys = [0] * 88

data = pd.read_csv('piano_keys_data.csv')
for index in range(len(data)):
    melody_keys = data.iloc[index].tolist()

    for key in range(88):
        # Handle melody track
        if melody_keys[key] == 1 and previous_melody_keys[key] == 0:
            # Note on
            track.append(Message('note_on', note=key + 21, velocity=64, time=0))
        elif melody_keys[key] == 0 and previous_melody_keys[key] == 1:
            # Note off
            track.append(Message('note_off', note=key + 21, velocity=64, time=0))

    track.append(Message('note_on', note=0, velocity=0, time=TICKS_PER_SNAPSHOT))

# Save the MIDI file
output_path = 'single_file_output/'
output_data_name = 'SF_output.mid'
if not os.path.exists(output_path):
    os.makedirs(output_path)

mid.save(f'{output_path}{output_data_name}')

print(f'MIDI file saved to {output_path}')