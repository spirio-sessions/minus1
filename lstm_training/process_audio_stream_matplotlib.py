import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

from lstm_training.extract_pitch import extract_pitch
def process_audio_stream_matplotlib(model, device, sequence_length, num_features):
    CHUNK = 1024  # Number of audio samples per frame
    RATE = 44100  # Sampling rate

    # Initialize pitch extractor and audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Initialize buffer for LSTM input
    buffer = np.zeros((sequence_length, num_features))

    # Initialize plot
    fig, ax = plt.subplots()
    x = np.arange(0, CHUNK)
    line, = ax.plot(x, np.random.rand(CHUNK))
    ax.set_ylim(-32768, 32767)

    def update_plot(frame):
        # Read audio data from the microphone
        data = stream.read(CHUNK)
        # Convert byte data to numpy array
        samples = np.frombuffer(data, dtype=np.int16)
        line.set_ydata(samples)

        # Extract pitch from audio data
        pitch = extract_pitch(data, sample_rate=RATE, buffer_size=CHUNK)  # Adjusted buffer size

        # Update buffer with new pitch data
        buffer[:-1] = buffer[1:]
        buffer[-1] = pitch

        # Prepare input for the model
        input_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict the left hand accompaniment
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        left_hand_output = output.cpu().numpy()

        # Print the prediction
        # print(f"Predicted left hand output: {left_hand_output}")

        return line,
    while True:
        ani = FuncAnimation(fig, update_plot, blit=True)
        plt.show()

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
