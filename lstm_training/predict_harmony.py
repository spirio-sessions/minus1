import numpy as np
import torch


def predict_harmony(model, melody, initial_harmony=None, max_len=50):

    # Check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    melody = np.expand_dims(melody, axis=0)  # Shape (1, sequence_length, input_size)
    melody_tensor = torch.tensor(melody, dtype=torch.float32).to(device)

    if initial_harmony is None:
        initial_harmony = np.zeros((melody.shape[0], max_len, melody.shape[2]))  # Shape (1, max_len, input_size)
    else:
        initial_harmony = np.expand_dims(initial_harmony, axis=0)  # Shape (1, sequence_length, input_size)

    harmony_tensor = torch.tensor(initial_harmony, dtype=torch.float32).to(device)
    harmony = []

    hidden = model.init_hidden(melody_tensor.size(0), device)

    # Von Felix: transformer -> inference -> inference_3 ANSCHAUEN.
    with torch.no_grad():
        for t in range(max_len):
            # Select the input sequence up to the current time step
            input_sequence = melody_tensor[:, :t+1, :]
            harmony_sequence = harmony_tensor[:, :t+1, :]

            # Concatenate the melody and harmony sequences along the feature dimension
            combined_input = torch.cat((input_sequence, harmony_sequence), dim=2)  # Shape (1, t+1, 2*input_size)

            # Predict the next note
            prediction, hidden = model(combined_input, hidden)
            next_note = prediction[0, -1, :].argmax().item()
            harmony.append(next_note)

            # Update the harmony tensor with the predicted note
            if t + 1 < max_len:
                harmony_tensor[0, t+1, next_note] = 1.0  # Assuming one-hot encoding for the notes


            """
            melody = np.expand_dims(melody, axis=0)  # Shape (1, sequence_length, input_size)
            melody_tensor = torch.tensor(melody, dtype=torch.float32).to(device)
    
            if initial_harmony is None:
                initial_harmony = np.zeros_like(melody)  # Shape (1, sequence_length, input_size)
            harmony_tensor = torch.tensor(initial_harmony, dtype=torch.float32).to(device)
    
            combined_input = torch.cat((melody_tensor, harmony_tensor), dim=2)  # Shape (1, sequence_length, 2*input_size)
    
            hidden = model.init_hidden(combined_input.size(0), device)
            harmony, hidden = model(combined_input, hidden)
            harmonies = harmony.cpu().numpy()
            """

    return harmony
