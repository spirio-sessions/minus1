import os
import torch


def save_model(save_path, save_parameters, model, num=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Determine the base file path
    base_file_path = os.path.join(save_path, f'lstm_0{num}')

    # Determine the model and parameter file paths
    model_file_path = f'../{base_file_path}.pt'
    parameter_file_path = f'../{base_file_path}.txt'

    if not os.path.exists(model_file_path):
        # Save the model
        torch.save(model.state_dict(), model_file_path)
        print(f'Model saved to {model_file_path}')

        # Save the parameters
        with open(parameter_file_path, 'w') as f:
            for param in save_parameters:
                f.write(f'{param}\n')
        print(f'Parameters saved to {parameter_file_path}')
    else:
        save_model(save_path, save_parameters, model, num + 1)
