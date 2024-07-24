import torch
from torch import nn
from tqdm import tqdm

from lstm_training.LSTMModel import LSTMModel
from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.prepare_dataset_dataloaders import prepare_dataset_dataloaders
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
num_layers = 4
OUTPUT_SIZE = 24
learning_rate = 0.0005
num_epochs = 20
batch_size = 64
seq_length = 256
stride = 64
databank = 'csv'
data_cap = 512




# Load melody and harmony from csv and can be caped
# melody, harmony = load_data_from_csv('csv')
data = load_data_from_csv(databank, data_cap)

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')

train_loader, val_loader = prepare_dataset_dataloaders(data, seq_length, stride, batch_size)

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

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        # batch => (128, 32, 24) -> (batch_size, seq_length, keyboard_size)
        batch = batch.to(device)
        inputs, targets = torch.split(batch, seq_length//2, dim=1)  # splits the sequence into two

        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0), device)

        # Forward pass
        outputs, hidden = model(inputs, hidden)

        # Reshape outputs and targets for CrossEntropyLoss
        outputs = outputs.reshape(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
        targets = targets.reshape(-1, targets.size(2))  # (batch_size * seq_length)
        targets = targets.argmax(dim=1)  # flattens it to (batch_size * sql_length, _)

        # Loss computation
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
        for batch in val_loader:
            # batch => (128, 32, 24) -> (batch_size, seq_length, keyboard_size)
            batch = batch.to(device)
            inputs, targets = torch.split(batch, seq_length//2, dim=1)  # splits the sequence into two

            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0), device)

            # Forward pass
            outputs, hidden = model(inputs, hidden)

            # Reshape outputs and targets for CrossEntropyLoss
            outputs = outputs.reshape(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
            targets = targets.reshape(-1, targets.size(2))  # (batch_size * seq_length)
            targets = targets.argmax(dim=1)  # flattens it to (batch_size * sql_length, _)

            # Loss computation
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
save_parameter = [INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE, learning_rate,
                  num_epochs, batch_size, seq_length, stride, databank, data_cap]
save_model('../04_finished_model/models', save_parameter, model)
