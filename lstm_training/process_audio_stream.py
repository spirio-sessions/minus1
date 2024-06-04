import numpy as np
import pyaudio
import torch

from lstm_training.extract_pitch import extract_pitch


def process_audio_stream(model, device, sequence_length, num_features):
    CHUNK = 1024  # Number of audio samples per frame
    RATE = 44100  # Sampling rate

    # Initialize pitch extractor and audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Initialize buffer for LSTM input
    buffer = np.zeros((sequence_length, num_features))
    # hidden = model.init_hidden(1)

    while True:
        data = stream.read(CHUNK)
        pitch = extract_pitch(data)  # Implement this function to extract pitch from audio data

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
        print(f"Predicted left hand output: {left_hand_output}")

# TODO: left_hand_output als linke Hand und pitch als rechte Hand kombinieren und MIDI erzeugen.
# TODO: Richtigen Pitchextractor noch einbauen.