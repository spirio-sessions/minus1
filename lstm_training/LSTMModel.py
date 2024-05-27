import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_data(melody_path, harmony_path):
    melody_df = pd.read_csv(melody_path)
    harmony_df = pd.read_csv(harmony_path)

    print("Melody DataFrame shape:", melody_df.shape)
    print("Harmony DataFrame shape:", harmony_df.shape)

    return melody_df.values, harmony_df.values


melody, harmony = load_data(
    'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\csv\\55Dive_melody.csv',
    'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\csv\\55Dive_harmony.csv'
)

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MelodyHarmonyDataset(Dataset):
    def __init__(self, melody, harmony):
        self.melody = melody
        self.harmony = harmony

    def __len__(self):
        return len(self.melody)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.melody[idx], dtype=torch.float32).unsqueeze(0),  # Add sequence_length dimension
            torch.tensor(self.harmony[idx], dtype=torch.float32)
        )

# Split data

melody_df = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\csv\\55Dive_melody.csv')
harmony_df = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\csv\\55Dive_harmony.csv')

# Split data
melody, harmony = melody_df.values, harmony_df.values

scaler = MinMaxScaler()
melody_scaled = scaler.fit_transform(melody)
harmony_scaled = scaler.fit_transform(harmony)

melody_train, melody_val, harmony_train, harmony_val = train_test_split(melody, harmony, test_size=0.2, random_state=42)
train_dataset = MelodyHarmonyDataset(melody_train, harmony_train)
val_dataset = MelodyHarmonyDataset(melody_val, harmony_val)

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for melodies, harmonies in train_loader:
    print("Batch melodies shape:", melodies.shape)  # Should be (batch_size, 1, 88)
    print("Batch harmonies shape:", harmonies.shape)  # Should be (batch_size, 88)
    break


import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=88, hidden_size=64, num_layers=2, output_size=88).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for melodies, harmonies in train_loader:
        melodies, harmonies = melodies.to(device), harmonies.to(device)

        outputs = model(melodies)
        loss = criterion(outputs, harmonies)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for melodies, harmonies in val_loader:
            melodies, harmonies = melodies.to(device), harmonies.to(device)

            outputs = model(melodies)
            loss = criterion(outputs, harmonies)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')


def predict_harmony(model, melody):
    model.eval()
    harmonies = []
    with torch.no_grad():
        for i in range(melody.shape[0]):
            single_melody = torch.tensor(melody[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            harmony = model(single_melody)
            harmonies.append(harmony.squeeze(0).cpu().numpy())
    return np.array(harmonies)


# Predict new melody
new_melody = pd.read_csv(
    'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\predict_melody\AFifthofBeethoven_melody.csv').values
predicted_harmony = predict_harmony(model, new_melody)

# Print different stats
print("Shape of Harmony")
print('- '*20)
print(predicted_harmony.shape)
print('- '*20)
print('- '*20)

print("First 5 lines of predicted harmony")
print('- '*20)
for i in range(5):
    print(predicted_harmony[i])
print('- '*20)
print('- '*20)

for i in range(5):
    plt.figure()
    plt.plot(predicted_harmony[i])
    plt.title(f'Predicted Harmony for Melody Row {i}')
    plt.show()

from sklearn.metrics import mean_squared_error

print("Mean-Square-Error")
print('- '*20)
# Assuming actual_harmony is available
actual_harmony = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\predict_melody\AFifthofBeethoven_melody.csv').values

mse = mean_squared_error(actual_harmony, predicted_harmony)
print(f'Mean Squared Error: {mse}')
print('- '*20)
print('- '*20)

import matplotlib.pyplot as plt

for i in range(5):
    plt.figure()
    plt.plot(predicted_harmony[i], label='Predicted')
    plt.plot(actual_harmony[i], label='Actual')
    plt.title(f'Harmony Prediction for Sample {i}')
    plt.legend()
    plt.show()

print("Compare with a Baseline")
print('- '*20)
baseline_prediction = np.mean(actual_harmony, axis=0)
baseline_mse = mean_squared_error(actual_harmony, np.tile(baseline_prediction, (actual_harmony.shape[0], 1)))
print(f'Baseline Mean Squared Error: {baseline_mse}')
print('- '*20)
print('- '*20)

print("Understand Scale of Data")
print('- '*20)
# Inspect the range and distribution of actual harmony values
print("Min value:", np.min(actual_harmony))
print("Max value:", np.max(actual_harmony))
print("Mean value:", np.mean(actual_harmony))
print("Standard deviation:", np.std(actual_harmony))
print('- '*20)

# Export to CSV

output_path = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\jazz_mlready_dataset\\small_batch\\predicted_melody\\predicted_csv.csv'
predicted_harmony_df = pd.DataFrame(predicted_harmony)
predicted_harmony_df.to_csv(output_path, index=False)
