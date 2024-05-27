import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from lstm_training.LSTMModel import LSTMModel
from lstm_training.MelodyHarmonyDataset import MelodyHarmonyDataset
from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.predict_harmony import predict_harmony
from lstm_training.print_results import print_results

# Load melody and harmony from csv
melody, harmony = load_data_from_csv('G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets'
                                     '\\jazz_mlready_dataset\\small_batch\\csv')

# Parameters
input_size = 88
hidden_size = 64
num_layers = 2
output_size = 88
learning_rate = 0.01
num_epochs = 5
batch_size = 32

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preparing data
melody_train, melody_val, harmony_train, harmony_val = train_test_split(melody, harmony, test_size=0.2, random_state=42)
train_dataset = MelodyHarmonyDataset(melody_train, harmony_train)
val_dataset = MelodyHarmonyDataset(melody_val, harmony_val)

# DataLoader
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
        melodies, harmonies = melodies.to(device), harmonies.to(device)

        outputs = model(melodies)
        loss = criterion(outputs, harmonies)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    loss = 0
    with torch.no_grad():
        for melodies, harmonies in val_loader:
            melodies, harmonies = melodies.to(device), harmonies.to(device)

            outputs = model(melodies)
            loss = criterion(outputs, harmonies)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Predict new melody
new_melody = pd.read_csv(
    'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\jazz_mlready_dataset\\small_batch'
    '\\predict_melody'
    '\\AgeOfAquarius_harmony.csv').values
predicted_harmony = predict_harmony(model, new_melody)

# Export to CSV
output_path = 'G:\\Schule\\Studium\\8. Semester\\Bachelor-Minus1\\minus1\\datasets\\jazz_mlready_dataset\\small_batch' \
              '\\predicted_melody\\'

predicted_harmony_df = pd.DataFrame(predicted_harmony)
new_melody_df = pd.DataFrame(new_melody)

new_melody_df.to_csv(output_path+"original_melody.csv", index=False)
predicted_harmony_df.to_csv(output_path+"predicted_harmony.csv", index=False)


print_results(predicted_harmony)
