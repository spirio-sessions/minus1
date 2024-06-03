import pandas as pd

from lstm_training.print_results import print_results

predicted_harmony = pd.read_csv('../datasets/maestro_v3_split/small_batch_lstm/predicted_leftH/predicted_harmony.csv').values

original_melody = pd.read_csv(
    '../datasets/maestro_v3_split/small_batch_lstm/original_validation/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1-split_rightH.csv').values
original_harmony = pd.read_csv(
    '../datasets/maestro_v3_split/small_batch_lstm/original_validation/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1-split_leftH.csv'
)

print_results(predicted_harmony, original_melody, original_harmony)


