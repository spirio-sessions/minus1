import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from lstm_training.SingleSequenceDataset import SingleSequenceDataset
from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.predict_outcome import predict_entire_song

"""
This script loads a model from the models directory and predicts a harmony from a given melody.
In the end it returns three CSV-files: 
    Both of the original melody/harmony set and the predicted harmony for further use.
"""

models_dir = 'models/experiments'
output_path = '../04_finished_model/models/experiments/results/'

model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
model_files = model_files[::-1]
context_length = 64

# List existing PNG files to check which models have already been processed
existing_pngs = [f for f in os.listdir(output_path) if f.endswith('_harmony_heatmap.png')]
processed_models = [os.path.splitext(f)[0].replace('_harmony_heatmap', '') for f in existing_pngs]

# Remove models that have already been processed
model_files = [f for f in model_files if os.path.splitext(f)[0] not in processed_models]

# validation_melody_name = 'validation/song_300_rightH.csv'
# validation_harmony_name = 'validation/song_300_leftH.csv'

validation_melody_name = 'validation/own_maria_rightH.csv'
validation_harmony_name = 'validation/own_maria_leftH.csv'

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
original_melody = pd.read_csv(validation_melody_name).values
original_harmony = pd.read_csv(validation_harmony_name).values
true_dataset = np.concatenate((original_harmony, original_melody), axis=1)

dataset = SingleSequenceDataset(true_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

original_melody_df = pd.DataFrame(original_melody)
original_harmony_df = pd.DataFrame(original_harmony)

original_melody_df.to_csv(output_path+"original_melody.csv", index=False)
print(f'{output_path}original_melody.csv was saved successfully!')
original_harmony_df.to_csv(output_path+"original_harmony.csv", index=False)
print(f'{output_path}original_harmony.csv was saved successfully!')


# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

counter: int = 0
# for model_file in tqdm(model_files, desc="Processing models and generating heatmaps", unit="Sequences"):
for model_file in model_files:
    counter = counter + 1
    print(f"Processing model {counter} of {len(model_files)+1}...")
    model_name = os.path.splitext(model_file)[0]  # Remove the .pt
    print(f"Processing model: {model_name}")

    # Load the model and parameters
    model, parameters = load_lstm_model('models/experiments', model_name, device)
    seq_length = int(parameters["seq_length"])

    # Predict song
    predicted_data = predict_entire_song(model, dataset, context_length, device)

    predicted_data_df = pd.DataFrame(predicted_data)
    predicted_harmony_df = predicted_data_df.iloc[:, :12].copy()
    predicted_harmony_df.iloc[:context_length, :] = 0

    predicted_data_df.to_csv(output_path+model_name+"_predicted_data.csv", index=False)
    predicted_harmony_df.to_csv(output_path+model_name+"_predicted_harmony.csv", index=False)

    print(f"Files for model {model_name} saved successfully!")

    # Create and save a heatmap of Predicted Harmony Data
    plt.figure(figsize=(20, 10))  # Adjust the size as necessary
    sns.heatmap(predicted_harmony_df, cmap='coolwarm', cbar_kws={'label': 'Probability of pressing the key'}, center=0.5, vmin=0, vmax=1)  # Adjust color map and limits based on your data
    plt.title('Heatmap of Predicted Harmony Data')
    plt.xlabel('Keys on piano, C -> Bb')
    plt.ylabel('Snapshot in MIDI')
    plt.savefig(f"{output_path}{model_name}_harmony_heatmap.png")
    plt.show()

    print(f"Heatmap for model {model_name} saved successfully!")
