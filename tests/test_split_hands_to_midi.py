from data_preperation.maestro_split_hands_into_midi import split_midi_tracks


input_folder = '../datasets/maestro-split-v3/small_batch_lstm/mid'
output_folder = '../datasets/maestro-split-v3/small_batch_lstm/mid_split'

split_midi_tracks(input_folder, output_folder)
