from data_preperation import dataset_snapshot
from data_preperation.dataset_snapshot import *

dataset_as_snapshots = dataset_snapshot.process_dataset("../datasets/jazz_mlready_dataset/small_batch/", .1)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

melody_harmony_dataset = extract_melody_and_harmony(filtered_dataset)
print_melody_harmony_dataset(melody_harmony_dataset)
export_melody_harmony_to_csv(melody_harmony_dataset, "../datasets/jazz_mlready_dataset/small_batch/csv")

