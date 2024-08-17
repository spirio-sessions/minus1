import json
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

   This function reads the model parameters from a JSON file and loads the
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
       - save_parameters (dict): A dictionary of model parameters loaded from the JSON file.

   Raises:
   FileNotFoundError: If the model or parameter file does not exist.

   Example:
   >>> model, parameters = load_lstm_model('/path/to/models', 'my_lstm_model', device='cuda')
   >>> print(model)
   LSTMModel(...)
   >>> print(parameters)
   {'INPUT_SIZE': 64, 'hidden_size': 128, ...}
   """
    model_file_path = f'{path}/{model_name}.pt'
    parameter_file_path = f'{path}/{model_name}.json'

    if os.path.exists(parameter_file_path):
        with open(parameter_file_path, 'r') as f:
            save_parameters = json.load(f)
    else:
        raise FileNotFoundError(f"No parameter file found at {parameter_file_path}")


    # Unpack parameters from the dictionary
    input_size = save_parameters.get("INPUT_SIZE")
    hidden_size = save_parameters.get("hidden_size")
    num_layers = save_parameters.get("num_layers")
    output_size = save_parameters.get("OUTPUT_SIZE")

    model = LSTMModel(int(input_size), int(hidden_size), int(num_layers), int(output_size)).to(device)

    # Load model state
    if os.path.exists(model_file_path):
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.eval()
    else:
        raise FileNotFoundError(f'No saved model found at {model_file_path}')

    print(f'Model loaded from {model_file_path}')
    print(f'Parameters loaded from {parameter_file_path}')

    return model, save_parameters
