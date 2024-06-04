import numpy as np
import aubio


def extract_pitch(data, sample_rate=44100, buffer_size=1024, hop_size=512):
    # Convert byte data to numpy array
    samples = np.frombuffer(data, dtype=aubio.float_type)

    # Create pitch detector
    pitch_detector = aubio.pitch("yin", buffer_size, hop_size, sample_rate)

    # Set pitch detector parameters
    pitch_detector.set_unit("midi")
    pitch_detector.set_silence(-40)

    # Compute pitch
    pitch = pitch_detector(samples)[0]

    # If pitch is found, return the MIDI pitch value, otherwise return None
    if pitch != 0:
        return int(pitch)
    else:
        return None
