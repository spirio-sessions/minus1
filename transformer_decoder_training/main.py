import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from data_preperation import dataset_snapshot
from dataset import create_dataloader
from transformer_decoder_only_model import TransformerDecoderModel
from data_model_specific_preperation import split_sequences

# Seed für Reproduzierbarkeit setzen
torch.manual_seed(42)
np.random.seed(42)

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Schritt 1: Daten laden und filtern
dataset_dir = "/home/falaxdb/Repos/minus1/datasets/maestro_v3_split/hands_split_into_seperate_midis"
data = dataset_snapshot.process_dataset_multithreaded(dataset_dir, 0.1)
filtered_data = dataset_snapshot.filter_piano_range(data)
split_data = split_sequences(filtered_data, max_len=1000)

# Datenaufteilung in Trainings-, Validierungs- und Testdatensätze
train_data, test_data = train_test_split(split_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(split_data, test_size=0.2, random_state=42)

# DataLoader erstellen
train_loader = create_dataloader(train_data, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_data, batch_size=32, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=32, shuffle=False)

# Schritt 2: Modell initialisieren
input_dim = 88  # Anzahl der Tasten eines Pianos
embed_dim = 256
nhead = 8
num_layers = 6
dim_feedforward = 512
output_dim = 88  # Gleiche Anzahl wie input_dim

model = TransformerDecoderModel(input_dim, embed_dim, nhead, num_layers, dim_feedforward, output_dim).to(device)

# Schritt 3: Trainingseinstellungen konfigurieren
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Schritt 4: Trainings- und Validierungsloop
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for melody, accompaniment in train_loader:
        melody, accompaniment = melody.to(device), accompaniment.to(device)
        optimizer.zero_grad()
        output = model(melody, accompaniment)
        loss = criterion(output, accompaniment)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validierungsschleife
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for melody, accompaniment in val_loader:
            melody, accompaniment = melody.to(device), accompaniment.to(device)
            output = model(melody, accompaniment)
            loss = criterion(output, accompaniment)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Bestes Modell speichern
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Schritt 5: Testen des Modells
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for melody, accompaniment in test_loader:
        melody, accompaniment = melody.to(device), accompaniment.to(device)
        output = model(melody, accompaniment)
        loss = criterion(output, accompaniment)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")
