import numpy as np
import mido
from mido import Message

from lstm_realtime_processing.generate_left_hand import generate_left_hand
from lstm_realtime_processing.midi_to_array import midi_to_array


def process_midi_message(msg, model, hidden, device, previous_harmony_keys, threshold):
    # Convert MIDI message to array
    input_array = midi_to_array(msg)

    # Generate left-hand accompaniment
    left_hand_array, hidden = generate_left_hand(input_array, model, hidden, device)

    # Apply threshold to the model output
    left_hand_array = np.where(left_hand_array > threshold, 1, 0)

    # Create MIDI messages for the left-hand accompaniment
    midi_messages = []
    for key in range(12):
        if left_hand_array[key] == 1 and previous_harmony_keys[key] == 0:
            # Note on
            midi_messages.append(Message('note_on', note=key + 21 + 36, velocity=64, time=0))
        elif left_hand_array[key] == 0 and previous_harmony_keys[key] == 1:
            # Note off
            midi_messages.append(Message('note_off', note=key + 21 + 36, velocity=64, time=0))

    return midi_messages, left_hand_array, hidden
