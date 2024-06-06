from data_preperation import dataset_snapshot
from data_preperation.csv.csv_loader import export_maestro_hands_to_csv
from data_preperation.filter_piano_range import filter_piano_range
from data_preperation.globals import INTERVAL

dataset_as_snapshots = dataset_snapshot.process_dataset('single_file_input', INTERVAL)

filtered_dataset = filter_piano_range(dataset_as_snapshots)

export_maestro_hands_to_csv(filtered_dataset, "G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\pipeline_lstm\\420_developer_tests_only\single_file_output")

print("Success")

