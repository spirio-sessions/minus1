from data_preperation import maestro_split_snapshot
from data_preperation.maestro_split_snapshot import *

dataset_as_snapshots = maestro_split_snapshot.process_dataset("../datasets/maestro-split-v3/small_batch_lstm/mid", 5)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

export_snapshots_to_csv(filtered_dataset, "../datasets/maestro-split-v3/small_batch_lstm/csv")

