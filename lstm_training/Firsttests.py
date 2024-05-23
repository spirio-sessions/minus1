import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mido
from mido import MidiFile, MidiTrack, Message

# Example dataset: list of (name, melody, harmony)
data = [
    ("song1", [60, 62, 64, 65, 67], [48, 50, 52, 53, 55]),
    ("song2", [67, 69, 71, 72, 74], [55, 57, 59, 60, 62]),
    # Add more data here
]

# Preprocess data
def preprocess(data):
    melodies = []
    harmonies = []
    for name, melody, harmony in data:
        melodies.append(melody)
        harmonies.append(harmony)

    X = []
    y = []

    for i in range(len(melodies)):
        for j in range(len(melodies[i]) - 1):
            X.append([melodies[i][j], harmonies[i][j]])
            y.append([melodies[i][j + 1], harmonies[i][j + 1]])

    X = np.array(X)
    y = np.array(y)

    return X, y

X, y = preprocess(data)

# Define a custom Dataset
class MusicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

dataset = MusicDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Expecting input of shape (batch, seq_len, input_size)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
        output = self.fc(lstm_out)
        return self.softmax(output)

input_size = 2
hidden_size = 128
output_size = 128

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs.unsqueeze(1))  # Ensure 3D input: (batch, seq_len, input_size)
        targets = targets[:, 0].long()  # Convert targets to long type for CrossEntropyLoss
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate music
def generate_sequence(model, seed, length):
    model.eval()
    sequence = []
    current = torch.tensor(seed, dtype=torch.float32).unsqueeze(0)  # Ensure 3D input: (batch, seq_len, input_size)
    for _ in range(length):
        output = model(current)
        next_note = torch.argmax(output, dim=1).item()
        sequence.append(next_note)
        next_input = torch.tensor([[next_note, 0]], dtype=torch.float32).unsqueeze(0)  # Ensure 3D input: (batch, seq_len, input_size)
        current = torch.cat((current[:, :, 1:], next_input), dim=1)
    return sequence

# Seed and length for generation
seed = [[60, 48]]
generated_sequence = generate_sequence(model, seed, 50)

# Convert to MIDI
def sequence_to_midi(sequence, filename="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note in sequence:
        track.append(Message('note_on', note=note, velocity=64, time=480))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    mid.save(filename)

sequence_to_midi(generated_sequence)

print("MIDI file generated: output.mid")
