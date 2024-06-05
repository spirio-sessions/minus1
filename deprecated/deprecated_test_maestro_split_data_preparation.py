from deprecated import deprecated_maestro_split_snapshot
from deprecated.deprecated_maestro_split_snapshot import *

INTERVAL = 0.05
dataset_as_snapshots = maestro_split_snapshot.process_dataset("../datasets/maestro_v3_split/maestro_hands_seperated_in_tracks",
                                                              INTERVAL, False, 40)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

export_snapshots_to_csv(filtered_dataset, "../datasets/maestro_v3_split/small_batch_lstm/csv")
