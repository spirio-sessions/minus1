import pandas as pd

from data_preperation.globals import original_melody_path, original_harmony_path
from lstm_training.print_results import print_results

predicted_harmony = pd.read_csv(
    '../../datasets/maestro_v3_split/small_batch_lstm/predicted_leftH/predicted_harmony.csv').values

original_melody = pd.read_csv(original_melody_path).values
original_harmony = pd.read_csv(original_harmony_path).values

print_results(predicted_harmony, original_melody, original_harmony)


