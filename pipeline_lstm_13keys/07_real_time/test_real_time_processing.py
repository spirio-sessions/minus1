import mido
import torch

from lstm_realtime_processing.autoregressive_process import autoregressive_process
from lstm_realtime_processing.process_midi_message import process_midi_message
from lstm_training.load_lstm_model import load_lstm_model

"""
This script loads a model and its parameters, takes the midi input/output and initiates an audio stream.
You may have to alter midi_input_name and midi_output_name after your physical midi input/output.
The process can be terminated using Strg+C.
The value 'threshold' determines at what threshold the one-hot-encoding probability actually plays the key or doesnt play it.
If the threshold is high, it only plays notes the model is really sure about.
If the threshold is low, it plays more notes, even tho the model is not really sure, if they fit.
This version uses the 12-key data model.
"""

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as device.')
model, parameters = load_lstm_model('../04_finished_model/models', 'realtime_70min_alldata', device)
model.eval()

# Unpack parameters
input_size = int(parameters[0])
hidden_size, num_layers, output_size, learning_rate, num_epochs, batch_size = parameters[1:]

# Threshold for note activation & temperature for randomness of autoregressive behaviour
threshold = 0.15
temperature = 1.0

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


print('\x1b[8;30;42m', "Listening for MIDI input... Press Strg+C to quit.", '\x1b[0m')

seed_sequence = [0] * 12  # Initial seed sequence

for msg in input_port:
    if msg.type in ['note_on', 'note_off']:
        midi_messages, previous_harmony_keys, hidden = process_midi_message(msg, model, hidden, device,
                                                                            previous_harmony_keys, threshold)
        """
        # Attempt to make it auto-regressiv
        next_note, hidden = autoregressive_process(model, seed_sequence[-input_size:], hidden, device, temperature)
        seed_sequence.append(next_note)
        seed_sequence = seed_sequence[-input_size:]
        midi_note = mido.Message('note_on', note=next_note)
        output_port.send(midi_note)
        """

        # Send generated MIDI messages to the output port
        for midi_msg in midi_messages:
            output_port.send(midi_msg)
