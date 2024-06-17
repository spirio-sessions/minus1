import pandas as pd

from lstm_training.print_results import print_results, printHeatmap

"""
This script takes the predicted harmony and both originals for statistik and inference.
The print includes:
    1. Mean-Square-Error
    2. Compare with baseline
    3. Scale of data with min, max, mean, std
    4. Plot of predicted_harmony
    5. Heatmap for key-probability in snapshot
    6. Heatmap for which key is getting played at what time
"""

# Load predicted data from MIDI
predicted_harmony = pd.read_csv('predicted_leftH/predicted_harmony.csv').values
original_melody = pd.read_csv('predicted_leftH/original_melody.csv').values
original_harmony = pd.read_csv('predicted_leftH/original_harmony.csv').values


print_results(predicted_harmony, original_melody, original_harmony)
printHeatmap(predicted_harmony, 0.5, 0, 1)

