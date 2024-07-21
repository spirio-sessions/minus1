import numpy as np
import torch

from lstm_training.LSTMModel import LSTMModel


def predict_harmony(model, melody, initial_harmony=None):

    # Check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        melody = np.expand_dims(melody, axis=0)  # Shape (1, sequence_length, input_size)
        melody_tensor = torch.tensor(melody, dtype=torch.float32).to(device)

        if initial_harmony is None:
            initial_harmony = np.zeros_like(melody)  # Shape (1, sequence_length, input_size)
        harmony_tensor = torch.tensor(initial_harmony, dtype=torch.float32).to(device)

        combined_input = torch.cat((melody_tensor, harmony_tensor), dim=2)  # Shape (1, sequence_length, 2*input_size)

        hidden = model.init_hidden(combined_input.size(0), device)
        harmony, hidden = model(combined_input, hidden)
        harmonies = harmony.cpu().numpy()

    return harmonies[0]  # Remove batch dimension before returning
