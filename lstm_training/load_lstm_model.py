import os
import torch

from lstm_training.LSTMModel import LSTMModel


def parse_line(line):
    try:
        return float(line.strip())
    except ValueError:
        return line.strip().strip('"')


def load_lstm_model(path, model_name, device='cpu'):
    """
    Load an LSTM model and its saved parameters from specified files.

    This function reads the model parameters from a text file and loads the
    corresponding trained LSTM model from a .pt file. The model is then moved
    to the specified device (CPU or GPU) and set to evaluation mode.

    Parameters:
    path (str): The directory path where the model and parameter files are stored.
    model_name (str): The base name of the model and parameter files (without extension).
    device (str): The device to which the model should be moved ('cpu' or 'cuda').
                  Default is 'cpu'.

    Returns:
    tuple: A tuple containing:
        - model (LSTMModel): The loaded LSTM model.
        - save_parameters (list): A list of model parameters loaded from the text file:
            [input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size]

    Raises:
    FileNotFoundError: If the model or parameter file does not exist.
    ValueError: If the parameter file does not contain the expected number of parameters.

    Example:
    >>> model, parameters = load_lstm_model('/path/to/models', 'my_lstm_model', device='cuda')
    >>> print(model)
    LSTMModel(...)
    >>> print(parameters)
    [64, 128, 2, 10, 0.001, 20, 32]
    """
    model_file_path = f'{path}/{model_name}.pt'
    parameter_file_path = f'{path}/{model_name}.txt'

    with open(parameter_file_path, 'r') as f:
        save_parameters = [parse_line(line) for line in f]


    # Unpack parameters
    input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size, seq_length, stride, databank, amount_data = save_parameters

    model = LSTMModel(int(input_size), int(hidden_size), int(num_layers), int(output_size)).to(device)

    # Load model state
    if os.path.exists(model_file_path):
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.eval()
    else:
        print(f'No saved model found at {model_file_path}')
        exit(1)

    print(f'Model loaded from {model_file_path}')
    print(f'Parameters loaded from {parameter_file_path}')

    return model, save_parameters
