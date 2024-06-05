import aubio
import numpy as np
import pyaudio
import torch

from lstm_training.extract_pitch import extract_pitch


def process_audio_stream(model, device, sequence_length, num_features):
    # Initialize pitch extractor and audio stream
    p = pyaudio.PyAudio()

    # open stream
    buffer_size = 1024
    pyaudio_format = pyaudio.paFloat32
    n_channels = 1
    samplerate = 44100
    stream = p.open(format=pyaudio_format,
                    channels=n_channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=buffer_size)

    # setup pitch
    tolerance = 0.1
    win_s = 4096  # fft size
    hop_s = buffer_size  # hop size
    pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    # Initialize buffer for LSTM input
    buffer = np.zeros((sequence_length, num_features))
    # hidden = model.init_hidden(1)

    while True:
        data = stream.read(buffer_size)
        pitch = extract_pitch(stream, pitch_o, buffer_size)  # Implement this function to extract pitch from audio data

        # Update buffer with new pitch data
        buffer[:-1] = buffer[1:]
        buffer[-1] = pitch

        # Prepare input for the model
        input_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict the left hand accompaniment
        model.eval()
        with torch.no_grad():
            # output, hidden = model(input_tensor, hidden)
            output = model(input_tensor)

        left_hand_output = output.cpu().numpy()

        # Print the prediction
        # print(f"Predicted left hand output: {left_hand_output}")


# TODO: left_hand_output als linke Hand und pitch als rechte Hand kombinieren und MIDI erzeugen.