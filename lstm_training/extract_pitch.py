import numpy as np
import pyaudio
import aubio


def extract_pitch(stream, pitch_o, buffer_size):
    audiobuffer = stream.read(buffer_size)
    signal = np.frombuffer(audiobuffer, dtype=np.float32)

    pitch = pitch_o(signal)[0]
    confidence = pitch_o.get_confidence()

    # Round the pitch to the nearest whole number
    rounded_pitch = round(pitch)

    # Create a list to represent the piano keys
    piano_keys = [0] * (109 - 21)  # [21:108]

    # If the rounded pitch is within the range of piano keys, set the corresponding index to 1
    if 21 <= rounded_pitch <= 108:
        piano_keys[rounded_pitch - 21] = 1

    print("{} / {} -> {}".format(pitch, confidence, piano_keys))

    return piano_keys
