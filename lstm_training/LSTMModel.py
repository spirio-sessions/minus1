import torch
from torch import nn


class LSTMModel(nn.Module):
    """
    A PyTorch implementation of an LSTM (Long Short-Term Memory) neural network.

    This class defines an LSTM model with a specified number of input features,
    hidden units, LSTM layers, and output features. The model includes an LSTM
    layer followed by a fully connected layer to map the hidden state output
    to the desired output size.

    Attributes:
    input_size (int): Number of input features.
    hidden_size (int): Number of hidden units in each LSTM layer.
    num_layers (int): Number of LSTM layers.
    output_size (int): Number of output features.

    Methods:
    forward(x):
        Defines the forward pass of the model.
        Initializes the hidden state and cell state, passes the input through
        the LSTM layers, and applies the fully connected layer to the last
        time step's output.

    Example:
    >>> model = LSTMModel(input_size=10, hidden_size=50, num_layers=2, output_size=1)
    >>> input_tensor = torch.randn(32, 5, 10)  # Batch size of 32, sequence length of 5, 10 features
    >>> output = model(input_tensor)
    >>> print(output.shape)
    torch.Size([32, 1])
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the LSTMModel with the specified parameters.

        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of LSTM layers.
        output_size (int): Number of output features.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # * 2

    def forward(self, x):
        """
      Defines the forward pass of the LSTMModel.

      Parameters:
      x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

      Returns:
      torch.Tensor: The output tensor of shape (batch_size, output_size).
      """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # * 2
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # * 2
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

    # TODO: Hidden-Layer als Verbesserung des LSTM Modells?
