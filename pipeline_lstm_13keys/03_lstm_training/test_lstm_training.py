import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lstm_training.LSTMModel import LSTMModel
from lstm_training.MelodyHarmonyDataset import MelodyHarmonyDataset
from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.save_model import save_model

"""
This script will train a LSTM-Model (long short-term memory) with the preprocessed CSV-files.
It uses several hyperparameter to finetune the model.
Right now it uses a MSE (mean-squared-error) and a custom music-theory as loss-function and Adam as optimizer.
It outputs a model.ht and a parameters.txt for further use.
"""

# Parameters
INPUT_SIZE = 24
hidden_size = 64
num_layers = 8  # 2
OUTPUT_SIZE = 24
learning_rate = 0.001
num_epochs = 10
batch_size = 128
seq_length = 32
databank = 'csv'
data_cap = 200




# Load melody and harmony from csv and can be caped
# melody, harmony = load_data_from_csv('csv')
melody, harmony = load_data_from_csv(databank, data_cap)

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')

# Preparing data
melody_train, melody_val, harmony_train, harmony_val = train_test_split(melody, harmony, test_size=0.2, random_state=42)
prep_dataset = MelodyHarmonyDataset(melody_train, harmony_train, seq_length)

# DataLoader
train_loader = DataLoader(prep_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(prep_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Model, loss function, optimizer
model = LSTMModel(INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE).to(device)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print('Starting training...')
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # rightHand = data[:, :12]  # First half of the data
        # leftHand = data[:, 12:].long()  # Second half of the data, converted for cross-entropy

        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0), device)

        # Forward pass
        outputs, hidden = model(inputs, hidden)  # Ensuring for always batched (batch_size, 1, feature_size)

        # Loss computation
        # CrossEntropyLoss expects (N, C) and (N), where C = number of classes
        outputs = outputs.view(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
        targets = targets.view(-1)  # (batch_size * seq_length)
        loss = criterion(outputs, targets)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Update model parameters
        optimizer.step()

        train_loss += loss.item()

    # Calculate and print average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # Validation step
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    # loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0), device)

            # Forward pass
            outputs, hidden = model(inputs, hidden)  # (batch_size, seq_length, feature_size)

            # Loss computation
            outputs = outputs.view(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
            targets = targets.view(-1)  # (batch_size * seq_length)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
save_parameter = [INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE, learning_rate,
                  num_epochs, batch_size, seq_length, databank, data_cap]
save_model('../04_finished_model/models', save_parameter, model)
