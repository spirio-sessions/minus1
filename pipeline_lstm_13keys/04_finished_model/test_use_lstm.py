import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm_training.SingleSequenceDataset import SingleSequenceDataset
from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.predict_outcome import predict_entire_song

"""
This script loads a model from the models directory and predicts a harmony from a given melody.
In the end it returns three CSV-files: 
    Both of the original melody/harmony set and the predicted harmony for further use.
"""

model_name = 'lstm_08'
validation_melody_name = 'validation/song_300_rightH.csv'
validation_harmony_name = 'validation/song_300_leftH.csv'

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and parameters
model, parameters = load_lstm_model('models', model_name, device)
seq_length = int(parameters[7])
context_length = seq_length // 2

# Load data
original_melody = pd.read_csv(validation_melody_name).values
original_harmony = pd.read_csv(validation_harmony_name).values
true_dataset = np.concatenate((original_harmony, original_melody), axis=1)

dataset = SingleSequenceDataset(true_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

# generated_tokens, predicted_harmony, input_seq = predict_outcome(model, dataset, seq_length, device)
# generated_tokens, predicted_harmony, input_seq = predict_outcome(model, dataset, seq_length, device)
predicted_data = predict_entire_song(model, dataset, context_length, device)


# Export to CSV
output_path = '../05_inference/predicted_leftH/'

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

predicted_data_df = pd.DataFrame(predicted_data)
predicted_harmony_df = predicted_data_df.iloc[:, :12].copy()
predicted_harmony_df.iloc[:768, :] = 0

original_melody_df = pd.DataFrame(original_melody)
original_harmony_df = pd.DataFrame(original_harmony)

original_melody_df.to_csv(output_path+"original_melody.csv", index=False)
print(f'{output_path}original_melody.csv was saved successfully!')
original_harmony_df.to_csv(output_path+"original_harmony.csv", index=False)
print(f'{output_path}original_harmony.csv was saved successfully!')

predicted_data_df.to_csv(output_path+"predicted_data.csv", index=False)
print(f'{output_path}predicted_data.csv was saved successfully!')
predicted_harmony_df.to_csv(output_path+"predicted_harmony.csv", index=False)
print(f'{output_path}predicted_harmony.csv was saved successfully!')
