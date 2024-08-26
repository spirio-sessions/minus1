import json
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
context_length = 64

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
model_files = model_files[::-1]

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

counter: int = 0
# Generate CSV if it is missing
for model_file in model_files:
    counter = counter + 1
    model_name = os.path.splitext(model_file)[0]  # Remove the .pt

    csv_path = os.path.join(output_path, model_name + "_predicted_data.csv")
    if not os.path.exists(csv_path):
        print(f"CSV for model {model_name} is missing. Generating CSV...")
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
        print(f"CSV files for model {model_name} of {len(model_files)+1} generated and saved successfully.")
    else:
        print(f"CSV for model {model_name} of {len(model_files)+1} already exists. Skipping CSV generation.")

counter = 0
for model_file in model_files:
    counter = counter + 1
    print(f"Processing model {counter} of {len(model_files)+1}...")
    model_name = os.path.splitext(model_file)[0]  # Remove the .pt
    print(f"Processing model: {model_name}")

    # Paths for PNG and JSON files
    png_path = os.path.join(output_path, model_name + "_harmony_heatmap.png")
    json_path = os.path.join(output_path, model_name + "_statistics.json")

    # Load CSV if it exists
    csv_path = os.path.join(output_path, model_name + "_predicted_harmony.csv")
    if os.path.exists(csv_path):
        predicted_harmony_df = pd.read_csv(csv_path)

        # Generate PNG if missing
        if not os.path.exists(png_path):
            # Create and save a heatmap of Predicted Harmony Data
            plt.figure(figsize=(20, 10))  # Adjust the size as necessary
            sns.heatmap(predicted_harmony_df, cmap='coolwarm', cbar_kws={'label': 'Probability of pressing the key'}, center=0.5, vmin=0, vmax=1)  # Adjust color map and limits based on your data
            plt.title(f'Heatmap of Predicted Harmony Data of {model_name}')
            plt.xlabel('Keys on piano, C -> Bb')
            plt.ylabel('Snapshot in MIDI')
            plt.savefig(f"{output_path}{model_name}_harmony_heatmap.png")
            plt.show()
            print(f"Heatmap for model {model_name} saved successfully!")
        else:
            print(f"PNG for model {model_name} already exists. Skipping PNG generation.")

        # Generate JSON if missing
        if not os.path.exists(json_path):
            stats = {'model_name': model_name,
                     'total_notes': predicted_harmony_df.sum().sum(),
                     'average_certainty': predicted_harmony_df.mean().mean(),
                     'highest_certainty': predicted_harmony_df.max().max(),
                     'lowest_certainty': predicted_harmony_df.min().min()
                     }

            # Ranking of notes from C to Bb based on average probability
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'Bb']
            note_averages = predicted_harmony_df.mean().tolist()
            stats['note_ranking'] = sorted(zip(note_names, note_averages), key=lambda x: x[1], reverse=True)

            # Save statistics to a JSON file
            with open(f"{output_path}{model_name}_statistics.json", 'w') as json_file:
                json.dump(stats, json_file, indent=4)
            print(f"Statistics for model {model_name} saved successfully!")
        else:
            print(f"JSON for model {model_name} already exists. Skipping JSON generation.")
    else:
        print(f"CSV for model {model_name} is missing. Cannot generate PNG or JSON.")


plt.figure(figsize=(20, 10))  # Adjust the size as necessary
sns.heatmap(original_harmony_df, cmap='coolwarm', cbar_kws={'label': 'Probability of pressing the key'}, center=0.5, vmin=0, vmax=1)  # Adjust color map and limits based on your data
plt.title('Heatmap of Predicted Harmony Data')
plt.xlabel('Keys on piano, C -> Bb')
plt.ylabel('Snapshot in MIDI')
plt.savefig(f"{output_path}{'original_harmony'}_harmony_heatmap.png")
plt.show()

plt.figure(figsize=(20, 10))  # Adjust the size as necessary
sns.heatmap(original_melody_df, cmap='coolwarm', cbar_kws={'label': 'Probability of pressing the key'}, center=0.5, vmin=0, vmax=1)  # Adjust color map and limits based on your data
plt.title('Heatmap of Predicted Harmony Data')
plt.xlabel('Keys on piano, C -> Bb')
plt.ylabel('Snapshot in MIDI')
plt.savefig(f"{output_path}{'original_melody'}_harmony_heatmap.png")
plt.show()
plt.close()

"""
# List existing PNG files to check which models have already been processed
existing_pngs = [f for f in os.listdir(output_path) if f.endswith('_harmony_heatmap.png')]
processed_models = [os.path.splitext(f)[0].replace('_harmony_heatmap', '') for f in existing_pngs]

# List existing JSON files to check which models have already been processed
existing_jsons = [f for f in os.listdir(output_path) if f.endswith('_statistics.json')]
processed_json_models = [os.path.splitext(f)[0].replace('_statistics', '') for f in existing_jsons]

# Remove models that have already been processed
model_files = [f for f in model_files if os.path.splitext(f)[0]
               not in processed_models and os.path.splitext(f)[0]
               not in processed_json_models]
"""
