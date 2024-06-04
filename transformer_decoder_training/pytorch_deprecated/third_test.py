import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded("../../datasets/temp", 0.1)
piano_range_dataset = dataset_snapshot.filter_piano_range(dataset_as_snapshots)

all_snapshots = []
for filename, snapshots in piano_range_dataset:
    for i in range(1, len(snapshots)):
        input_seq = snapshots[:i]
        target_seq = snapshots[i:i + 1]
        all_snapshots.append((input_seq, target_seq))

train_data, test_data = train_test_split(all_snapshots, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f'Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}')


class PianoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return inputs_padded, targets_padded


batch_size = 32  # Reduced batch size
train_loader = DataLoader(PianoDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(PianoDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(PianoDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


class MusicTransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(MusicTransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        memory = self.embedding(memory)
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        output = self.transformer_decoder(tgt, memory)
        output = output.permute(1, 0, 2)
        return self.output_layer(output)


input_dim = 88
hidden_dim = 256
num_layers = 4
num_heads = 4
dropout = 0.1

model = MusicTransformerDecoder(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, accumulation_steps=2):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(targets, inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_targets, val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        # Print average loss for each epoch
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


train_model(model, train_loader, val_loader)


def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            test_outputs = model(test_targets, test_inputs)
            test_loss += criterion(test_outputs, test_targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')


evaluate_model(model, test_loader)


def generate_accompaniment(model, melody, max_length=100):
    model.eval()
    accompaniment = torch.zeros_like(melody).to(device)
    melody = melody.to(device)

    for i in range(max_length):
        with torch.no_grad():
            output = model(accompaniment, melody)
            accompaniment[:, i, :] = (output[:, i, :] > 0).float()

    return accompaniment


# Example melody (replace with actual data)
melody = torch.zeros((1, 100, 88))
accompaniment = generate_accompaniment(model, melody)
