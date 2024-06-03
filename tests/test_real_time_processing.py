import torch

from lstm_training.load_data_from_csv import load_data_from_csv
from lstm_training.load_lstm_model import load_lstm_model
from lstm_training.process_audio_stream import process_audio_stream


# Load melody and harmony from csv
melody, harmony = load_data_from_csv('../datasets/maestro_v3_split/small_batch_lstm/csv')


# Parameters
num_features = 88
sequence_length = 5

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')

model, parameters = load_lstm_model('lstm_04', device)

# Unpack parameters
input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size = parameters

# Call the function to start processing
process_audio_stream(model, device, sequence_length, num_features)
