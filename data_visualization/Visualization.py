import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_losses(train_losses: list, val_losses: list, file_path: str):
    """
    Function to plot training and validation losses and save the plot to a specified file path.

    Parameters:
    - train_losses (list): List of training loss values.
    - val_losses (list): List of validation loss values.
    - file_path (str): Path where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
    plt.savefig(file_path)  # Save the plot to the specified file path
    plt.show()


def plot_list_of_tensors_heatmaps(tensors: list, title='Probabilities Heatmaps'):
    """
    Function to plot the probabilities of a list of 2D tensors as heatmaps.

    Parameters:
    - tensors (list of torch.Tensor): List of 2D tensors with values between 0 and 1.
    - title (str): Title for the entire plot.
    """
    num_tensors = len(tensors)

    # Create a figure with subplots for each tensor
    fig, axes = plt.subplots(1, num_tensors, figsize=(12 * num_tensors, 6))
    if num_tensors == 1:
        axes = [axes]  # Ensure axes is iterable when there is only one subplot

    for i, tensor in enumerate(tensors):
        probabilities = tensor.numpy()

        sns.heatmap(probabilities, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                    xticklabels=range(probabilities.shape[1]), yticklabels=range(probabilities.shape[0]),
                    linewidths=.5, linecolor='gray', annot_kws={"size": 12}, ax=axes[i])

        axes[i].set_xlabel('Vector Index')
        axes[i].set_ylabel('Sequence Index')
        axes[i].set_title(f'Tensor {i + 1}')
        axes[i].tick_params(axis='x', rotation=0)  # Ensure the x-tick labels are horizontal
        axes[i].tick_params(axis='y', rotation=0)  # Ensure the y-tick labels are horizontal

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
