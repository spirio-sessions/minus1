from data_preperation import dataset_snapshot

dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded("../datasets/midi_test_files", 1)

dataset_snapshot.print_dataset(dataset_as_snapshots)