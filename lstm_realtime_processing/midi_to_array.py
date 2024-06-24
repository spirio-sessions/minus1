import numpy as np


def midi_to_array(msg):
    """
    Convert a MIDI message to a 12-element array representation.
    """
    array = np.zeros(12)
    if msg.type == 'note_on' and msg.velocity > 0:
        note = msg.note % 12
        array[note] = 1
    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        note = msg.note % 12
        array[note] = 0
    return array
