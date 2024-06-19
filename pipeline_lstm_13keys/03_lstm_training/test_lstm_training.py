import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from lstm_training.LSTMModel import LSTMModel
from lstm_training.MelodyHarmonyDataset import MelodyHarmonyDataset
from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.save_model import save_model
from lstm_training.music_theory_loss import MusicTheoryLoss

"""
This script will train a LSTM-Model (long short-term memory) with the preprocessed CSV-files.
It uses several hyperparameter to finetune the model.
Right now it uses a MSE (mean-squared-error) as loss-function and Adam as optimizer.
It outputs a model.ht and a parameters.txt for further use.
"""


# Load melody and harmony from csv and can be caped
# melody, harmony = load_data_from_csv('csv')
melody, harmony = load_data_from_csv('csv', 40)

# Parameters
INPUT_SIZE = 12
hidden_size = 64
num_layers = 2
OUTPUT_SIZE = 12
learning_rate = 0.01
num_epochs = 2
batch_size = 60

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')

# Preparing data
melody_train, melody_val, harmony_train, harmony_val = train_test_split(melody, harmony, test_size=0.2, random_state=42)
train_dataset = MelodyHarmonyDataset(melody_train, harmony_train)
val_dataset = MelodyHarmonyDataset(melody_val, harmony_val)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
model = LSTMModel(INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE).to(device)
# criterion = nn.MSELoss()
criterion = MusicTheoryLoss(alpha=0.5, beta=1.0)  # Alpha equals weight of MSE, beta weight of custom loss-function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print('Starting training...')
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for melodies, harmonies in train_loader:
        melodies, harmonies = melodies.to(device), harmonies.to(device)

        # Forward pass
        outputs = model(melodies)

        # Loss computation
        loss = criterion(outputs, harmonies, melodies)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Validation step
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for melodies, harmonies in val_loader:
            melodies, harmonies = melodies.to(device), harmonies.to(device)

            # Forward pass
            outputs = model(melodies)

            # Loss computation
            loss = criterion(outputs, harmonies, melodies)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
save_parameter = [INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE, learning_rate, num_epochs, batch_size]
save_model('../04_finished_model/models', save_parameter, model)
