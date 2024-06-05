import numpy as np


def extract_pitch(stream, pitch_o, buffer_size):
    audiobuffer = stream.read(buffer_size)
    signal = np.frombuffer(audiobuffer, dtype=np.float32)
    pitch = pitch_o(signal)[0]
    return pitch

# TODO: Needs further testing with real frequencies. Shows signs of errors
