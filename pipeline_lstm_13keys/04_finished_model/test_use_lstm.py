import os

import pandas as pd
import torch

from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.predict_harmony import predict_harmony

"""
This script loads a model from the models directory and predicts a harmony from a given melody.
In the end it returns two CSV-files with the original melody and the predicted harmony for further use.
"""


# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and parameters
model, parameters = load_lstm_model('models', 'first_realtime', device)

# Predict new melody
# original_melody = pd.read_csv('validation/validation_melody.csv').values
# original_harmony = pd.read_csv('validation/validation_harmony.csv').values
original_melody = pd.read_csv('validation/song_300_rightH.csv').values
original_harmony = pd.read_csv('validation/song_300_leftH.csv').values
predicted_harmony = predict_harmony(model, original_melody)

# Export to CSV
output_path = '../05_inference/predicted_leftH/'

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

predicted_harmony_df = pd.DataFrame(predicted_harmony)
original_melody_df = pd.DataFrame(original_melody)
original_harmony_df = pd.DataFrame(original_harmony)

original_melody_df.to_csv(output_path+"original_melody.csv", index=False)
print(f'{output_path}original_melody.csv was saved successfully!')

predicted_harmony_df.to_csv(output_path+"predicted_harmony.csv", index=False)
print(f'{output_path}predicted_harmony.csv was saved successfully!')

original_harmony_df.to_csv(output_path+"original_harmony.csv", index=False)
print(f'{output_path}original_harmony.csv was saved successfully!')
