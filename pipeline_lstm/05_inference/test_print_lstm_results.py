import pandas as pd

from lstm_training.print_results import print_results

predicted_harmony = pd.read_csv(
    'predicted_leftH/predicted_harmony.csv').values

original_melody = pd.read_csv('../04_finished_model/validation/validation_melody.csv').values
original_harmony = pd.read_csv('../04_finished_model/validation/validation_harmony.csv').values

print_results(predicted_harmony, original_melody, original_harmony)

