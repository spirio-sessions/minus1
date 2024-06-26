from data_preperation import dataset_snapshot
from data_preperation.csv.csv_loader import export_maestro_hands_to_csv_transpose_to_each_key
from data_preperation.globals import INTERVAL

"""
This script is the second script of the pipeline.
It uses the split-midi set and transforms it into snapshots in a specific interval to a 12keys array.
In the end the dataset will be exported to the dataformat CSV for training.
This in particular is a special version which uses "transpose_to_each_key" to multiply data and shows the model the 
importance of intervals
"""

dataset_as_snapshots = dataset_snapshot.process_dataset_12keys('mid-split', INTERVAL, amount=300)

export_maestro_hands_to_csv_transpose_to_each_key(dataset_as_snapshots, "../03_lstm_training/csv_transposed")
