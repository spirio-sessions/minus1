from datetime import datetime
import torch
from torch import nn
from tqdm import tqdm

from lstm_training.LSTMModel import LSTMModel
from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.music_theory_loss import MusicTheoryLoss
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
hidden_size = 512
num_layers = 4
OUTPUT_SIZE = 24
learning_rate = 0.0001
num_epochs = 15
batch_size = 256
seq_length = 8
stride = 1
databank = 'csv'
data_cap = 0




# Load melody and harmony from csv and can be caped
# melody, harmony = load_data_from_csv('csv')
data = load_data_from_csv(databank, data_cap)

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')

train_loader, val_loader = prepare_dataset_dataloaders(data, seq_length, stride, batch_size)

# Model, loss function, optimizer
model = LSTMModel(INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE).to(device)

criterion = nn.BCEWithLogitsLoss()
# criterion = MusicTheoryLoss(alpha=0.5, beta=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

# Training loop
print('Starting training...')
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    start_time = datetime.now()

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
        targets = targets.reshape(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
        # targets = targets.argmax(dim=1)  # flattens it to (batch_size * sql_length, _)
        # Normale Cross-entropy ohne Argmax probieren, da er nur in BinaryCrossEntropy richtig ist.

        # Loss computation
        loss = criterion(outputs, targets)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameters
        optimizer.step()

        train_loss += loss.item()

    # Calculate and print average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

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
            targets = targets.reshape(-1, OUTPUT_SIZE)  # (batch_size * seq_length, OUTPUT_SIZE)
            # targets = targets.argmax(dim=1)  # flattens it to (batch_size * sql_length, _)

            # Loss computation
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print the losses for the epoch
    elapsed_time = datetime.now() - start_time
    print(f"\n{'='*50}")
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Time: {elapsed_time}")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"{'='*50}\n")

# Save the parameter for json
save_parameters = {
    "INPUT_SIZE": INPUT_SIZE,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "OUTPUT_SIZE": OUTPUT_SIZE,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "seq_length": seq_length,
    "stride": stride,
    "databank": databank,
    "data_cap": data_cap,
    "train_loss": avg_train_loss,
    "val_loss": val_loss,
    "train_losses_list": train_losses,
    "val_losses_list": val_losses
}
# Save the trained model
save_model('../04_finished_model/models/experiments', save_parameters, model, train_losses, val_losses)
