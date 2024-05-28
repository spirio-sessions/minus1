import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data_preperation import dataset_snapshot
import numpy as np
import math
from tqdm import tqdm  # Fortschrittsbalken
import time  # Für Zeitmessung
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision

# Prüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datenvorbereitung
dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded(
    "../datasets/temp", 0.1)

# Liste von Tupeln (Dateiname und Numpy-Array von Snapshots mit variierender Länge)
piano_range_dataset = dataset_snapshot.filter_piano_range(dataset_as_snapshots)

# Alle Snapshots extrahieren
all_snapshots = []
for filename, snapshots in piano_range_dataset:
    for i in range(1, len(snapshots)):
        input_seq = snapshots[:i]
        target_seq = snapshots[i:i + 1]
        all_snapshots.append((input_seq, target_seq))

# Datenaufteilung in Trainings-, Validierungs- und Test-Sets
train_data, test_data = train_test_split(all_snapshots, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f'Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}')


# Dataset Klasse
class PianoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


# Collate Funktion
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return inputs_padded, targets_padded


# DataLoader erstellen
train_dataset = PianoDataset(train_data)
val_dataset = PianoDataset(val_data)
test_dataset = PianoDataset(test_data)

# Erhöhung der Batch-Größe
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8,
                        pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8,
                         pin_memory=True)


# Funktion zur Generierung der Positionskodierung
def generate_positional_encoding(seq_len, d_model):
    pos_enc = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return pos_enc


# Transformer-Decoder Modell
class MusicTransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(MusicTransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, tgt, memory):
        seq_len_tgt = tgt.size(1)
        seq_len_mem = memory.size(1)
        pos_enc_tgt = generate_positional_encoding(seq_len_tgt, self.embedding.out_features).to(tgt.device)
        pos_enc_mem = generate_positional_encoding(seq_len_mem, self.embedding.out_features).to(memory.device)

        tgt = self.embedding(tgt) + pos_enc_tgt
        memory = self.embedding(memory) + pos_enc_mem
        tgt = tgt.permute(1, 0, 2)  # Change to (seq_len, batch, feature)
        memory = memory.permute(1, 0, 2)  # Change to (seq_len, batch, feature)
        output = self.transformer_decoder(tgt, memory)
        output = output.permute(1, 0, 2)  # Change back to (batch, seq_len, feature)
        return self.output_layer(output)


# Parameter
input_dim = 88
hidden_dim = 256  # Reduzierte Größe für weniger Speicherverbrauch
num_layers = 4  # Weniger Schichten für weniger Speicherverbrauch
num_heads = 4  # Weniger Köpfe für weniger Speicherverbrauch
dropout = 0.1

model = MusicTransformerDecoder(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)


# Training Funktion
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, accumulation_steps=2):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # Für Mixed Precision

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        start_time = time.time()  # Zeitmessung starten
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with autocast():  # Mixed Precision
                outputs = model(targets, inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / (i + 1):.4f}')

        end_time = time.time()  # Zeitmessung beenden
        epoch_duration = end_time - start_time
        print(f'Time for epoch {epoch + 1}: {epoch_duration:.2f} seconds')

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device, non_blocking=True), val_targets.to(device,
                                                                                                   non_blocking=True)
                val_outputs = model(val_targets, val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


# Modelltraining
train_model(model, train_loader, val_loader)


# Modellbewertung
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.to(device, non_blocking=True), test_targets.to(device,
                                                                                                   non_blocking=True)
            test_outputs = model(test_targets, test_inputs)
            test_loss += criterion(test_outputs, test_targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')


# Modellbewertung
evaluate_model(model, test_loader)


# Echtzeit-Generierung
def generate_accompaniment(model, melody, max_length=100):
    model.eval()
    accompaniment = torch.zeros_like(melody).to(device)
    melody = melody.to(device)

    for i in range(max_length):
        with torch.no_grad():
            output = model(accompaniment, melody)
            accompaniment[:, i, :] = (output[:, i, :] > 0).float()

    return accompaniment


# Beispiel-Melodie (du musst sie mit deinen echten Daten implementieren)
melody = torch.zeros((1, 100, 88))  # Placeholder
accompaniment = generate_accompaniment(model, melody)
