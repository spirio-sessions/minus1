from data_preperation import dataset_snapshot
from Deprecated.deprecated_dataset_snapshot import *
from data_preperation.csv.csv_loader import export_maestro_hands_to_csv
from data_preperation.globals import INTERVAL

"""
This script is the second script of the pipeline.
It uses the split-midi set and transforms it into snapshots in a specific interval.
After that it gets filtered from MIDI-scale to piano-scale, by reducing the list via [21:109].
In the end the dataset will be exported to the dataformat CSV for training.
"""

dataset_as_snapshots = dataset_snapshot.process_dataset('mid-split', INTERVAL)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

# TODO: Doesn't work yet! Gains 20 instead of 40 tuple
export_maestro_hands_to_csv(filtered_dataset, "../03_lstm_training/csv")

# TODO: Mehr Doc-Strings