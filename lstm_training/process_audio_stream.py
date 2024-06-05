import aubio
import numpy as np
import pandas as pd
import pyaudio
import torch
import csv

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
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_size).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_size).to(device))

    pitch_data = []
    predicted_data = []

    while True:
        try:
            pitch = extract_pitch(stream, pitch_o, buffer_size)
            # confidence = pitch_o.get_confidence()

            rounded_pitch = round(pitch)
            piano_keys = [0] * num_features
            if 21 <= rounded_pitch <= 108:
                piano_keys[rounded_pitch - 21] = 1

            pitch_data.append(piano_keys)

            # Update buffer with new pitch data
            buffer[:-1] = buffer[1:]
            buffer[-1] = pitch

            # Prepare input for the model
            input_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict the left hand accompaniment
            model.eval()
            with torch.no_grad():
                output, hidden = model(input_tensor, hidden)
                # output = model(input_tensor)

            left_hand_output = output.cpu().numpy()
            predicted_data.append(left_hand_output)

            # Print the prediction
            print(f"Predicted left hand output: {left_hand_output}")

            hidden = tuple(h.detach() for h in hidden)

        except KeyboardInterrupt:
            print("*** Ctrl+C pressed, exiting")

            # Save and print pitch_data
            piano_data_df = pd.DataFrame(pitch_data)
            piano_data_list = piano_data_df.values
            with open('pitch_data.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(piano_data_list)
            print(f"Data written to pitch_data.csv")

            # Save and print predicted_data
            predicted_data_df = pd.DataFrame(predicted_data)
            predicted_data_list = predicted_data_df.values
            # TODO: OUTPUTS 1,88 instead of 88, which results in an error
            # Write the list of arrays to a CSV file
            with open('predicted_data.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(predicted_data_list)

            print("Data written to predicted_data.csv")
            break
