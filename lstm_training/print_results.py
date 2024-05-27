import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def print_results(predicted_harmony):
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
    for i in range(5):
        plt.figure()
        plt.plot(predicted_harmony[i])
        plt.title(f'Predicted Harmony for Melody Row {i}')
        plt.show()
    print("Mean-Square-Error")
    print('- ' * 20)
    # Assuming actual_harmony is available
    actual_harmony = pd.read_csv(
        'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\jazz_mlready_dataset\\small_batch'
        '\\predict_melody\\AFifthofBeethoven_melody.csv').values
    mse = mean_squared_error(actual_harmony, predicted_harmony)
    print(f'Mean Squared Error: {mse}')
    print('- ' * 20)
    print('- ' * 20)
    for i in range(5):
        plt.figure()
        plt.plot(predicted_harmony[i], label='Predicted')
        plt.plot(actual_harmony[i], label='Actual')
        plt.title(f'Harmony Prediction for Sample {i}')
        plt.xlabel('Keys on piano')
        plt.ylabel('probability of pressing (One-Hot-Encoding)')
        plt.legend()
        plt.show()
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
