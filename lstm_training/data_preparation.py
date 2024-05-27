import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# Data Preparation
class MelodyHarmonyDataset(Dataset):
    def __init__(self, melody, harmony):
        self.melody = melody
        self.harmony = harmony

    def __len__(self):
        return len(self.melody)

    def __getitem__(self, idx):
        return torch.tensor(self.melody[idx], dtype=torch.float32), torch.tensor(self.harmony[idx], dtype=torch.float32)


def load_data_from_csv(directory):
    melody_data = []
    harmony_data = []

    melody_files = [f for f in os.listdir(directory) if f.endswith('_melody.csv')]
    for melody_file in melody_files:
        base_filename = melody_file.replace('_melody.csv', '')
        harmony_file = base_filename + '_harmony.csv'

        melody_df = pd.read_csv(os.path.join(directory, melody_file))
        harmony_df = pd.read_csv(os.path.join(directory, harmony_file))

        melody_data.append(melody_df.values)
        harmony_data.append(harmony_df.values)

    return np.concatenate(melody_data), np.concatenate(harmony_data)


melody, harmony = load_data_from_csv(
    'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\jazz_mlready_dataset\\small_batch\\csv')

# Model_definition
import torch.nn as nn


import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h_0, c_0))

        # Decode the hidden state of t


# Hyperparameters
input_size = 128
hidden_size = 64
num_layers = 2
output_size = 128
learning_rate = 0.001
num_epochs = 50
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preparing data
melody_train, melody_val, harmony_train, harmony_val = train_test_split(melody, harmony, test_size=0.2, random_state=42)
train_dataset = MelodyHarmonyDataset(melody_train, harmony_train)
val_dataset = MelodyHarmonyDataset(melody_val, harmony_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for melodies, harmonies in train_loader:
        melodies = melodies.to(device)
        harmonies = harmonies.to(device)

        outputs = model(melodies)
        loss = criterion(outputs, harmonies)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for melodies, harmonies in val_loader:
            melodies = melodies.to(device)
            harmonies = harmonies.to(device)

            outputs = model(melodies)
            loss = criterion(outputs, harmonies)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')



# Prediction in Real-time
def predict_harmony(model, melody):
    model.eval()
    with torch.no_grad():
        melody = torch.tensor(melody, dtype=torch.float32).unsqueeze(0).to(device)  # Ensure the input is batched
        harmony = model(melody)
        return harmony.squeeze(0).cpu().numpy()


new_melody = pd.read_csv('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\datasets\jazz_mlready_dataset\small_batch\csv').values
predicted_harmony = predict_harmony(model, new_melody)
print(predicted_harmony)

