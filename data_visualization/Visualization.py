import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator


from plotly.subplots import make_subplots
from pygments.lexers import go


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


def visualize_heatmaps_plotly(tensor_list, n_cols=4):
    """
    Visualizes a list of tensors as a grid of heatmaps using Plotly.

    Parameters:
        tensor_list (list of torch.Tensor): List of tensors to visualize.
        n_cols (int): Number of columns in the grid. Adjust based on the number of tensors.
    """
    # Determine the number of rows needed
    n_rows = (len(tensor_list) + n_cols - 1) // n_cols

    # Prepare subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'Tensor {i + 1}' for i in range(len(tensor_list))])

    for i, tensor in enumerate(tensor_list):
        # Convert tensor to numpy array
        np_array = tensor.numpy()

        # Determine row and column in the subplot grid
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Add heatmap to the subplot
        heatmap = go.Heatmap(z=np_array, colorscale='Viridis', zmin=0, zmax=1)
        fig.add_trace(heatmap, row=row, col=col)

    # Update layout
    fig.update_layout(height=300 * n_rows, width=300 * n_cols, title_text="Heatmaps of Tensors", showlegend=False)
    fig.show()

