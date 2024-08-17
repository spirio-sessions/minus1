import os
import torch
from matplotlib import pyplot as plt

from lstm_training.LSTMModel import LSTMModel


def save_model(save_path, save_parameters, model, num_epochs, train_losses, val_losses, num=0):
    """
    Save a trained LSTM model and its parameters to specified files.

    This function saves the state dictionary of a given LSTM model and its associated
    parameters to a specified directory. If a file with the same name already exists,
    the function increments the number suffix and tries again to avoid overwriting.

    Parameters:
    save_path (str): The directory path where the model and parameter files will be saved.
    save_parameters (list): A list of parameters to save alongside the model. Typically includes
                            model configuration and training parameters.
    model (torch.nn.Module): The trained LSTM model to be saved.
    num (int, optional): A number suffix for the filenames to avoid overwriting. Default is 0.

    Returns:
    None

    Example:
    >>> model = LSTMModel(input_size=10, hidden_size=50, num_layers=2, output_size=1)
    >>> save_parameters = [10, 50, 2, 1, 0.001, 20, 32]
    >>> save_path = './saved_models'
    >>> save_model(save_path, save_parameters, model)
    Model saved to ../saved_models/lstm_00.pt
    Parameters saved to ../saved_models/lstm_00.txt
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Determine the base file path
    base_file_path = os.path.join(save_path, f'lstm_0{num}')

    # Determine the model and parameter file paths
    model_file_path = f'{base_file_path}.pt'
    parameter_file_path = f'{base_file_path}.txt'
    plot_file_path = f'{base_file_path}.png'

    if not os.path.exists(model_file_path):
        # Save the model
        torch.save(model.state_dict(), model_file_path)
        print(f'Model saved to {model_file_path}')

        # Save the parameters
        with open(parameter_file_path, 'w') as f:
            for param in save_parameters:
                f.write(f'{param}\n')
        print(f'Parameters saved to {parameter_file_path}')

        # Save the plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.show()
        plt.savefig(plot_file_path)
        print(f'Plot saved to {plot_file_path}')
    else:
        save_model(save_path, save_parameters, model, num_epochs, train_losses, val_losses, num + 1)


# Save plot
save_dir = '../04_finished_model/models/experiments'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

plot_path = os.path.join(save_dir, 'training_validation_loss.png')
