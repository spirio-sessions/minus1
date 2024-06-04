from data_preperation import dataset_snapshot
from Deprecated.deprecated_dataset_snapshot import *
from data_preperation.csv.csv_loader import export_maestro_hands_to_csv
from data_preperation.globals import INTERVAL

dataset_as_snapshots = dataset_snapshot.process_dataset("../../datasets/maestro_v3_split/small_batch_lstm/mid_split", INTERVAL)
filtered_dataset = filter_piano_range(dataset_as_snapshots)

# print_dataset(filtered_dataset)

# TODO: Doesn't work yet! Gains 20 instead of 40 tuple
export_maestro_hands_to_csv(filtered_dataset, "../datasets/maestro_v3_split/small_batch_lstm/csv")

# TODO: Mehr Doc-Strings