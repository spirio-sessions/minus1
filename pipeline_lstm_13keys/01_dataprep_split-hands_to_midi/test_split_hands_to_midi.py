from data_preperation.maestro_split_hands_into_midi import split_midi_tracks

"""
This script is the first script of the pipeline.
It takes all MIDI from a specific directory and outputs a split version of it for each seperated track.
The amount of tracks can be limited by passing a False flag and the amount of MIDIs after that
"""

input_folder = '../../datasets/maestro_v3_split/maestro_hands_seperated_in_tracks'
output_folder = '../02_dataprep_midi_to_csv/mid-split'


split_midi_tracks(input_folder, output_folder)  # Uses all MIDI-tracks
# split_midi_tracks(input_folder, output_folder, False, 40)  # Limited to 40 MIDI-tracks