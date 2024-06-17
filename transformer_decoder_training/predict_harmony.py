import torch
import numpy as np
from data_preperation import dataset_snapshot_tracks_as_midi_files
from transformer_decoder_only_model import TransformerDecoderModel

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modell initialisieren und laden
input_dim = 88  # Anzahl der Tasten eines Pianos
embed_dim = 256
nhead = 8
num_layers = 6
dim_feedforward = 512
output_dim = 88  # Gleiche Anzahl wie input_dim
max_len = 1000  # Maximale Länge der Eingabesequenzen

model = TransformerDecoderModel(input_dim, embed_dim, nhead, num_layers, dim_feedforward, output_dim).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Schritt 1: Daten laden und filtern
dataset_dir = "/home/falaxdb/Repos/minus1/datasets/inference_tests/original"
data = dataset_snapshot_tracks_as_midi_files.process_dataset(dataset_dir, 0.1)
filtered_data = dataset_snapshot_tracks_as_midi_files.filter_piano_range(data)

def normalize_melody(melody):
    return (melody - np.min(melody)) / (np.max(melody) - np.min(melody))

# Beispiel-Melodie auswählen und normalisieren
example_melody = normalize_melody(filtered_data[0][0])  # select right hand of first track
print("melody shape:", example_melody.shape)

def create_src_mask(src_len, max_len):
    mask = torch.zeros((src_len, src_len)).type(torch.bool)
    mask[:src_len, :src_len] = 1
    return mask.to(device)

# Funktion zur Vorhersage der Begleitung
def predict_accompaniment(melody, max_len=1000):
    model.eval()
    melody = np.array(melody)

    if len(melody) > max_len:
        melody = melody[:max_len]
    elif len(melody) < max_len:
        padding = np.zeros((max_len - len(melody), 88))
        melody = np.vstack((melody, padding))

    src_mask = create_src_mask(len(melody), max_len).to(device)

    melody_tensor = torch.tensor(melody, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(melody_tensor, melody_tensor, src_mask=src_mask).squeeze(0).cpu().numpy()

    print("Logits:", logits)  # Ausgabe der Rohlogits

    # Sigmoid-Funktion anwenden
    probabilities = 1 / (1 + np.exp(-logits))

    # Schwellenwert anwenden
    accompaniment = (probabilities > 0.5).astype(int)

    return accompaniment

# Melodie eingeben und Begleitung vorhersagen
predicted_accompaniment = predict_accompaniment(example_melody, max_len=1000)

print("Predicted Accompaniment:", predicted_accompaniment)
print("Predicted Accompaniment shape:", predicted_accompaniment.shape)
