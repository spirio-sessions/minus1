import mido
import numpy as np
import torch
from termcolor import colored
from mido import Message

from lstm_realtime_processing.generate_left_hand import generate_left_hand
from lstm_realtime_processing.midi_to_array import midi_to_array
from lstm_realtime_processing.process_midi_message import process_midi_message
from lstm_training.load_lstm_model import load_lstm_model

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')
model, parameters = load_lstm_model('../04_finished_model/models', 'new_lossF_all_notes_0025', device)
model.eval()

# Unpack parameters
input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size = parameters

# Threshold for note activation
threshold = 0.25

# MIDI constants
TICKS_PER_BEAT = 480  # Standard MIDI ticks per beat
TEMPO = 500000  # microseconds per beat, equivalent to 120 BPM
TICKS_PER_SNAPSHOT = int(TICKS_PER_BEAT * (0.5 / (60 / 120)))  # for 120 BPM

# Initial states of the keys
previous_melody_keys = [0] * 12
previous_harmony_keys = [0] * 12

# Initialize hidden state
hidden = model.init_hidden(1, device)

# MIDI input and output ports
print("Your MIDI inputs are:", mido.get_input_names(), "and outputs:", mido.get_output_names())
midi_input_name = 'USB MIDI Interface 0'
midi_output_name = 'USB MIDI Interface 1'
try:
    input_port = mido.open_input(midi_input_name)
    output_port = mido.open_output(midi_output_name)
except OSError:
    print('\x1b[0;30;41m', 'ERROR: Please input a functional MIDI-Name', '\x1b[0m')
    exit()
else:
    print("Using input:", midi_input_name, "and output:", midi_output_name)


print("Listening for MIDI input...")

for msg in input_port:
    if msg.type in ['note_on', 'note_off']:
        midi_messages, previous_harmony_keys, hidden = process_midi_message(msg, model, hidden, device,
                                                                            previous_harmony_keys, threshold)

        # Send generated MIDI messages to the output port
        for midi_msg in midi_messages:
            output_port.send(midi_msg)




