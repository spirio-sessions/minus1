from data_preperation import dataset_snapshot
from data_preperation.dataset_snapshot import *
from data_preperation.globals import INTERVAL

dataset_as_snapshots = dataset_snapshot.process_dataset("../datasets/maestro_v3_split/small_batch_lstm/mid_split",
                                                        INTERVAL, False, 40)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

# print_dataset(filtered_dataset)

export_maestro_hands_to_csv(filtered_dataset, "../datasets/maestro_v3_split/small_batch_lstm/csv")
