from data_preperation.maestro_split_hands_into_midi import split_midi_tracks


input_folder = '../datasets/maestro_v3_split/maestro_hands_seperated_in_tracks'
output_folder = '../datasets/maestro_v3_split/small_batch_lstm/mid_split'

split_midi_tracks(input_folder, output_folder, False, 20)
