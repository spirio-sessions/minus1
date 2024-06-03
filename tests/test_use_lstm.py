import os

import pandas as pd
import torch

from data_preperation.globals import original_melody_path
from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.predict_harmony import predict_harmony
from lstm_training.print_results import print_results

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and parameters
model, parameters = load_lstm_model('lstm_00', device)


# Predict new melody
original_melody = pd.read_csv(original_melody_path).values
predicted_harmony = predict_harmony(model, original_melody)


# Export to CSV
output_path = '../datasets/maestro_v3_split/small_batch_lstm/predicted_leftH/'

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

predicted_harmony_df = pd.DataFrame(predicted_harmony)
new_melody_df = pd.DataFrame(original_melody)

new_melody_df.to_csv(output_path+"original_melody.csv", index=False)


# Print results
predicted_harmony_df.to_csv(output_path+"predicted_harmony.csv", index=False)
original_harmony = pd.read_csv(
    '../datasets/maestro_v3_split/small_batch_lstm/original_validation/MIDI-Unprocessed_02_R1_2009_01-02_ORIG_MID'
    '--AUDIO_02_R1_2009_02_R1_2009_01_WAV-split_leftH.csv'
)
print_results(predicted_harmony, original_melody, original_harmony)
