import os

from lstm_training.LSTMModel import LSTMModel

import torch


def load_lstm_model(model_name, device='cpu'):
    model_file_path = f'../lstm_training/saved_models/{model_name}.pt'
    parameter_file_path = f'../lstm_training/saved_models/{model_name}.txt'

    with open(parameter_file_path, 'r') as f:
        save_parameters = [float(line.strip()) for line in f]

    # Unpack parameters
    input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size = save_parameters

    # Initiate model
    model = LSTMModel(int(input_size), int(hidden_size), int(num_layers), int(output_size)).to(device)

    # Load model state
    if os.path.exists(model_file_path):
        print(f'Model loaded from {model_file_path}')
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.eval()
    else:
        print(f'No saved model found at {model_file_path}')
        exit(1)

    print(f'Model loaded from {model_file_path}')
    print(f'Parameters loaded from {parameter_file_path}')

    return model, save_parameters
