import numpy as np
import torch

from lstm_training.LSTMModel import LSTMModel


def predict_harmony(model, melody):
    """
    Predict harmony sequences for given melody sequences using a trained LSTM model.

    This function takes a trained LSTM model and a set of melody sequences, and
    predicts the corresponding harmony sequences. The function ensures that the
    model is in evaluation mode and performs predictions without computing gradients.

    Parameters:
    model (torch.nn.Module): The trained LSTM model for predicting harmonies.
    melody (numpy.ndarray): A numpy array of shape (num_samples, sequence_length, input_size)
                            containing the melody sequences to predict harmonies for.

    Returns:
    numpy.ndarray: A numpy array of shape (num_samples, sequence_length, output_size)
                   containing the predicted harmony sequences.

    Example:
    >>> model = LSTMModel(input_size=10, hidden_size=50, num_layers=2, output_size=10)
    >>> melody_data = np.random.rand(100, 50, 10)  # 100 samples, sequence length of 50, 10 features
    >>> harmonies = predict_harmony(model, melody_data)
    >>> print(harmonies.shape)
    (100, 50, 10)
    """

    # Check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    harmonies = []
    with torch.no_grad():
        for i in range(melody.shape[0]):
            # Initialize hidden state
            hidden = model.init_hidden(melody.shape[0], device, unbatched=True)
            single_melody = torch.tensor(melody[i], dtype=torch.float32).unsqueeze(0).to(device)
            harmony, hidden = model(single_melody, hidden)
            harmonies.append(harmony.squeeze(0).cpu().numpy())
    return np.array(harmonies)
