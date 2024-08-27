import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
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


def calculate_note_ranking_diff(original_stats, model_stats, predicted_harmony_df, threshold=0.7):
    original_ranking = {note: rank for rank, (note, _) in enumerate(original_stats['note_ranking'])}
    model_ranking = {note: rank for rank, (note, _) in enumerate(model_stats['note_ranking'])}

    note_ranking_diff = 0

    for note in original_ranking:
        # Get the count of how often each note is played above the threshold
        note_play_count = (predicted_harmony_df[note_names.index(note)] >= threshold).sum()

        # Calculate penalty based on how far the note's rank is from the original
        rank_diff = abs(original_ranking[note] - model_ranking[note])

        # Penalize more if lower-ranked notes are played often
        note_ranking_diff += rank_diff * note_play_count

    return note_ranking_diff


def calculate_distinct_notes(df, threshold=0.7, min_duration=3):
    """
    Combine notes that are played with a high probability for an extended period.

    Parameters:
    - df: DataFrame with rows representing time and columns representing keys
    - threshold: Probability above which a note is considered "played"
    - min_duration: Minimum number of consecutive time points that qualify as a "long" duration

    Returns:
    - combined_notes_count: The number of distinct notes after combining
    """
    combined_notes_count = 0

    for column in df.columns:
        # Identify where the note is considered "played" based on the threshold
        played = df[column] >= threshold

        # Group consecutive "played" instances
        groups = (played != played.shift()).cumsum()

        # Evaluate the groups
        distinct_notes = 0
        for group, sub_df in df.groupby(groups):
            if sub_df[column].max() >= threshold and len(sub_df) >= min_duration:
                # This group of notes qualifies as a single "combined" note
                distinct_notes += 1

        combined_notes_count += distinct_notes

    return combined_notes_count


def calculate_composite_score(stats, distinct_notes, average_certainty_weight=100,
                              highest_certainty_weight=500, lowest_certainty_weight=500,
                              note_ranking_diff_weight=50, distinct_notes_weight=5):
    # Composite score considering distinct notes
    score_average_certainty_diff = stats['average_certainty_diff'] * average_certainty_weight
    score_highest_certainty_diff = stats['highest_certainty_diff'] * highest_certainty_weight
    score_lowest_certainty_diff = stats['lowest_certainty_diff'] * lowest_certainty_weight
    score_note_ranking_diff = stats['note_ranking_diff'] * note_ranking_diff_weight
    score_distinct_notes = distinct_notes * distinct_notes_weight

    return score_average_certainty_diff \
        + score_highest_certainty_diff \
        + score_note_ranking_diff \
        + score_lowest_certainty_diff \
        - score_distinct_notes


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
# Step 1: Generate missing CSV files
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
        print(f"CSV files for model {model_name} number {counter} of {len(model_files)+1} generated and saved successfully.")
    else:
        print(f"CSV for model {model_name} number {counter} of {len(model_files)+1} already exists. Skipping CSV generation.")

# Step 2: Generate PNG and JSON files (including original melody and harmony)

original_json_paths = {
    "original_harmony": os.path.join(output_path, "original_harmony_statistics.json"),
    "original_melody": os.path.join(output_path, "original_melody_statistics.json"),
}

for model_name in ["original_harmony", "original_melody"] + [os.path.splitext(f)[0] for f in model_files]:

    # Paths for PNG and JSON files
    png_path = os.path.join(output_path, model_name + "_harmony_heatmap.png")
    json_path = os.path.join(output_path, model_name + "_statistics.json")
    csv_path = os.path.join(output_path, model_name + "_predicted_harmony.csv")

    if model_name == "original_harmony":
        predicted_harmony_df = original_harmony_df
    elif model_name == "original_melody":
        predicted_harmony_df = original_melody_df
    else:
        if os.path.exists(csv_path):
            predicted_harmony_df = pd.read_csv(csv_path)
        else:
            continue

    # Generate PNG if missing
    if not os.path.exists(png_path):
        # Create and save a heatmap of Predicted Harmony Data
        plt.figure(figsize=(20, 10))  # Adjust the size as necessary
        sns.heatmap(predicted_harmony_df, cmap='coolwarm', cbar_kws={'label': 'Probability of pressing the key'}, center=0.5, vmin=0, vmax=1)  # Adjust color map and limits based on your data
        plt.title(f'Heatmap of Predicted Harmony Data of {model_name}')
        plt.xlabel('Keys on piano, C -> Bb')
        plt.ylabel('Snapshot in MIDI')
        plt.savefig(png_path)
        plt.show()
        print(f"Heatmap for model {model_name} saved successfully!")

    # Generate JSON if missing
    if not os.path.exists(json_path):
        stats = {'model_name': model_name,
                 'average_certainty': float(predicted_harmony_df.mean().mean()),
                 'highest_certainty': float(predicted_harmony_df.max().max()),
                 'lowest_certainty': float(predicted_harmony_df.min().min())
                 }

        # Ranking of notes from C to Bb based on average probability
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'Bb']
        note_averages = predicted_harmony_df.mean().tolist()
        stats['note_ranking'] = sorted(zip(note_names, note_averages), key=lambda x: x[1], reverse=True)

        # Calculate distinct notes and composite score
        distinct_notes = calculate_distinct_notes(predicted_harmony_df, threshold=0.7, min_duration=3)
        stats['distinct_notes'] = distinct_notes

    # Save statistics to a JSON file
        with open(f"{output_path}{model_name}_statistics.json", 'w') as json_file:
            json.dump(stats, json_file, indent=4)
        print(f"Statistics for model {model_name} saved successfully!")

# Step 3: Generate and compare statistics
original_json_path = os.path.join(output_path, "original_harmony_statistics.json")
with open(original_json_path, 'r') as f:
    original_stats = json.load(f)

comparison_results = []

# Compare each model's stats to the original harmony stats
for model_file in tqdm(model_files):
    model_name = os.path.splitext(model_file)[0]
    json_path = os.path.join(output_path, model_name + "_statistics.json")

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            model_stats = json.load(f)

        comparison = {
            'model_name': model_name,
            'average_certainty_diff': abs(model_stats['average_certainty'] - original_stats['average_certainty']),
            'highest_certainty_diff': abs(model_stats['highest_certainty'] - original_stats['highest_certainty']),
            'lowest_certainty_diff': abs(model_stats['lowest_certainty'] - original_stats['lowest_certainty']),
            'distinct_notes': model_stats['distinct_notes']
        }

        # Calculate note_ranking_diff by comparing the rankings of each note
        original_ranking = {note: rank for rank, (note, _) in enumerate(original_stats['note_ranking'])}
        model_ranking = {note: rank for rank, (note, _) in enumerate(model_stats['note_ranking'])}
        note_ranking_diff = sum(abs(original_ranking[note] - model_ranking[note]) for note in original_ranking)
        comparison['note_ranking_diff'] = note_ranking_diff

        # Calculate the composite score now that all diffs are available
        comparison['composite_score'] = calculate_composite_score(comparison, comparison['distinct_notes'])

        comparison_results.append(comparison)

# Sort by composite score (lower is better)
comparison_results = sorted(comparison_results, key=lambda x: x['composite_score'])

# Save the comparison results to a JSON file
comparison_output_path = os.path.join(output_path, "model_comparisons.json")

with open(comparison_output_path, 'w') as f:
    json.dump(comparison_results, f, indent=4)
    f.flush()  # Ensure all data is written to the file
    os.fsync(f.fileno())  # Ensure data is flushed to disk
print(f"Comparison results saved successfully to {comparison_output_path}")
