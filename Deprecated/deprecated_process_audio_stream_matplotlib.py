import aubio
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import torch


def process_audio_stream_matplotlib(model, device, sequence_length, num_features):
    buffer_size = 1024
    pyaudio_format = pyaudio.paFloat32
    n_channels = 1
    samplerate = 44100

    # Initialize pitch extractor and audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio_format, channels=n_channels, rate=samplerate, input=True, frames_per_buffer=buffer_size)

    # Setup Aubio pitch detection
    tolerance = 0.8
    win_s = 4096
    hop_s = buffer_size
    pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    # Initialize buffer for LSTM input
    buffer = np.zeros((sequence_length, num_features))

    # Initialize plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 87, 88)  # 88 keys on a piano
    bars = ax.bar(x, np.zeros(88), align='center')
    ax.set_ylim(0, 1)  # Set y-axis limits to 0 and 1
    ax.set_xlim(0, 87)  # Set x-axis limits to 88 keys

    def update_plot(frame):
        try:
            # Read audio data from the microphone
            audiobuffer = stream.read(buffer_size)
            # Convert byte data to numpy array and normalize to range [0, 1]
            signal = np.frombuffer(audiobuffer, dtype=np.float32)

            # Extract pitch from audio data
            pitch = pitch_o(signal)[0]  # Aubio pitch detection
            confidence = pitch_o.get_confidence()



            # Normalize pitch to range [0, 1] if confidence is high enough
            if confidence > 0.8:
                pitch_index = int(pitch)
                normalized_pitch = pitch / 88.0  # Normalize pitch to range [0, 1]
            else:
                pitch_index = 0
                normalized_pitch = 0

            print("{} / {}".format(normalized_pitch, confidence))
            # Update buffer with new pitch data
            buffer[:-1] = buffer[1:]
            buffer[-1, pitch_index] = normalized_pitch

            # Prepare input for the model
            input_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict the left hand accompaniment
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            left_hand_output = output.cpu().numpy()

            # Update bars
            for bar, h in zip(bars, buffer[-1]):
                bar.set_height(h)

            # Print the prediction
            # print(f"Predicted left hand output: {left_hand_output}")

            return bars,
        except Exception as e:
            print(f"An error occurred: {e}")
            return bars,

    while True:
        # ani = FuncAnimation(fig, update_plot, blit=True)
        # plt.show()
        update_plot(0)

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
