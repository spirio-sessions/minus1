import os
import torch
import pandas as pd

from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.predict_harmony import predict_harmony

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and parameters
model, parameters = load_lstm_model('../04_finished_model/models', 'lstm_03', device)

# Predict new melody
original_melody = pd.read_csv('single_file_output/song_1_rightH.csv').values
predicted_harmony = predict_harmony(model, original_melody)

# Export to CSV
output_path = 'single_file_output/'

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

predicted_harmony_df = pd.DataFrame(predicted_harmony)

predicted_harmony_df.to_csv(output_path+"predicted_harmony.csv", index=False)
print(f'{output_path}predicted_harmony.csv was saved successfully!')