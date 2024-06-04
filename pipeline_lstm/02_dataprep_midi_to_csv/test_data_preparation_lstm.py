from data_preperation import dataset_snapshot
from data_preperation.csv.csv_loader import export_maestro_hands_to_csv
from data_preperation.globals import INTERVAL
from data_preperation.maestro_split_snapshot import filter_piano_range

"""
This script is the second script of the pipeline.
It uses the split-midi set and transforms it into snapshots in a specific interval.
After that it gets filtered from MIDI-scale to piano-scale, by reducing the list via [21:109].
In the end the dataset will be exported to the dataformat CSV for training.
"""

dataset_as_snapshots = dataset_snapshot.process_dataset('mid-split', INTERVAL)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

export_maestro_hands_to_csv(filtered_dataset, "../03_lstm_training/csv")

# TODO: Mehr Doc-Strings