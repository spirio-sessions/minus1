import csv

import pyaudio
import sys
import numpy as np
import aubio

# initialize pyaudio
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
"""
if len(sys.argv) > 1:
    # record 5 seconds
    output_filename = sys.argv[1]
    record_duration = 5  # exit 1
    outputsink = aubio.sink(sys.argv[1], samplerate)
    total_frames = 0
else:
    # run forever
    outputsink = None
    record_duration = None
"""

record_duration = 5  # seconds
outputsink = None
total_frames = 0

# setup pitch
tolerance = 0.8
win_s = 4096  # fft size
hop_s = buffer_size  # hop size
pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

# Prepare to store the piano keys data
piano_keys_data = []

print("*** starting recording")
while True:
    try:
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

        piano_keys_data.append(piano_keys)

        if outputsink:
            outputsink(signal, len(signal))

        if record_duration:
            total_frames += len(signal)
            if record_duration * samplerate < total_frames:
                break
    except KeyboardInterrupt:
        print("*** Ctrl+C pressed, exiting")
        break

print("*** done recording")
stream.stop_stream()
stream.close()
p.terminate()

# Save the piano keys data to a CSV file
with open('piano_keys_data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Frame'] + [f'Key_{i}' for i in range(21, 109)])
    for frame_idx, keys in enumerate(piano_keys_data):
        csvwriter.writerow([frame_idx] + keys)

print("*** piano keys data saved to piano_keys_data.csv")
