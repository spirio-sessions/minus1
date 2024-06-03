import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from sklearn.metrics import mean_squared_error


def print_results(predicted_harmony, actual_melody, actual_harmony):
    # Print different stats
    print("Shape of Harmony")
    print('- ' * 20)
    print(predicted_harmony.shape)
    print('- ' * 20)
    print('- ' * 20)

    print("First 5 lines of predicted harmony")
    print('- ' * 20)
    for i in range(5):
        print(predicted_harmony[i])
    print('- ' * 20)
    print('- ' * 20)

    print("Mean-Square-Error")
    print('- ' * 20)
    mse = mean_squared_error(actual_harmony, predicted_harmony)
    print(f'Mean Squared Error: {mse}')
    print('- ' * 20)
    print('- ' * 20)

    print("Compare with a Baseline")
    print('- ' * 20)
    baseline_prediction = np.mean(actual_harmony, axis=0)
    baseline_mse = mean_squared_error(actual_harmony, np.tile(baseline_prediction, (actual_harmony.shape[0], 1)))
    print(f'Baseline Mean Squared Error: {baseline_mse}')
    print('- ' * 20)
    print('- ' * 20)

    print("Understand Scale of Data")
    print('- ' * 20)
    # Inspect the range and distribution of actual harmony values
    print("Min value:", np.min(actual_harmony))
    print("Max value:", np.max(actual_harmony))
    print("Mean value:", np.mean(actual_harmony))
    print("Standard deviation:", np.std(actual_harmony))
    print('- ' * 20)

    for i in range(5):
        plt.figure()
        plt.plot(predicted_harmony[i])
        plt.title(f'Predicted Harmony for Melody Row {i}')
        plt.show()

    # Create a heatmap
    plt.figure(figsize=(20, 10))  # Adjust the size as necessary
    sns.heatmap(predicted_harmony, cmap='coolwarm', center=0, vmin=0, vmax=1)  # Adjust color map and limits based on your data
    plt.title('Heatmap of Predicted Harmony Data')
    plt.xlabel('Keys on piano')
    plt.ylabel('probability of pressing (One-Hot-Encoding)')
    plt.show()


    # Create a second heatmap
    # Define the boundaries and colors for the custom colormap
    boundaries = [0, 0.03, 1]
    colors = ["#0096FF", "#00008b", "#FF474C", "#8b0000"]

    # Create a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Create a norm to map data values to the colormap
    norm = BoundaryNorm(boundaries, custom_cmap.N, extend='both')

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(predicted_harmony, cmap=custom_cmap, norm=norm, cbar=True)
    plt.show()
