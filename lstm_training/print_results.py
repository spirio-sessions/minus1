import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score


def print_results(predicted_harmony, actual_melody, actual_harmony):
    # Create the 'pictures' directory if it doesn't exist
    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    # Determine the next prefix number for saving files
    existing_files = [f for f in os.listdir('pictures') if f.endswith('.png')]
    if existing_files:
        latest_file = max(existing_files)
        latest_prefix = int(latest_file.split('_')[0])
        prefix = f"{latest_prefix + 1:02d}_"
    else:
        prefix = "00_"

    # Calculate Mean Squared Error
    mse = mean_squared_error(actual_harmony, predicted_harmony)
    print(f"Mean Squared Error: {mse}")
    print('- ' * 20)

    # Calculate Baseline Mean Squared Error
    baseline_prediction = np.mean(actual_harmony, axis=0)
    baseline_mse = mean_squared_error(actual_harmony, np.tile(baseline_prediction, (actual_harmony.shape[0], 1)))
    print(f"Baseline Mean Squared Error: {baseline_mse}")
    print('- ' * 20)

    # Scale of Data
    print("Scale of Actual Harmony Data")
    print(f"Min value: {np.min(actual_harmony)}")
    print(f"Max value: {np.max(actual_harmony)}")
    print(f"Mean value: {np.mean(actual_harmony)}")
    print(f"Standard deviation: {np.std(actual_harmony)}")
    print('- ' * 20)

    # Plot and save Predicted Harmony for Melody
    plt.figure()
    plt.plot(predicted_harmony[0])
    plt.title('Predicted Harmony for Melody')
    plt.savefig(os.path.join('pictures', f'{prefix}predicted_harmony_for_melody.png'))
    plt.show()

    # Create and save a heatmap of Predicted Harmony Data
    plt.figure(figsize=(20, 10))  # Adjust the size as necessary
    sns.heatmap(predicted_harmony, cmap='coolwarm', center=0.05, vmin=0, vmax=0.1)  # Adjust color map and limits based on your data
    plt.title('Heatmap of Predicted Harmony Data')
    plt.xlabel('Keys on piano')
    plt.ylabel('Probability of pressing (One-Hot-Encoding)')
    plt.savefig(os.path.join('pictures', f'{prefix}heatmap_predicted_harmony.png'))
    plt.show()

    # Create and save a second heatmap with a custom colormap
    boundaries = [0, 0.03, 0.05, 1]
    colors = ["#0096FF", "#00008b", "#FF474C", "#FFF74C", "#8b0000"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    norm = BoundaryNorm(boundaries, custom_cmap.N, extend='both')

    plt.figure(figsize=(10, 8))
    sns.heatmap(predicted_harmony, cmap=custom_cmap, norm=norm, cbar_kws={'label': 'Probability of pressing the key'})
    plt.title('Custom Heatmap of Predicted Harmony Data')
    plt.xlabel('Keys on piano')
    plt.ylabel('Snapshot in MIDI')
    plt.savefig(os.path.join('pictures', f'{prefix}custom_heatmap_predicted_harmony.png'))
    plt.show()

    # Additional Statistics
    predicted_labels = np.argmax(predicted_harmony, axis=1)
    actual_labels = np.argmax(actual_harmony, axis=1)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print('- ' * 20)