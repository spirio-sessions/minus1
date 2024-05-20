from data_preperation import dataset_snapshot

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

## Get dataset ##
dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded("../datasets/test2", 0.1)
piano_range_dataset = dataset_snapshot.filter_piano_range(dataset_as_snapshots)

## Turn data into batches ##
def prepare_sequences(dataset_as_snapshots, seq_length=10):
    inputs, targets = [], []
    for _, snapshots in dataset_as_snapshots:
        for i in range(len(snapshots) - seq_length):
            seq_in = snapshots[i:i + seq_length]
            seq_out = snapshots[i + seq_length]
            inputs.append(seq_in)
            targets.append(seq_out)
    return np.array(inputs), np.array(targets)

# Prepare data
# split data into sequences for inputs and targets, which are the snapshots directly after the sequence
inputs, targets = prepare_sequences(piano_range_dataset)
X_train, X_temp, y_train, y_temp = train_test_split(inputs, targets, test_size=0.2, random_state=42) # split for training and temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # split temp for test and validation

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, y_test.shape)

## Define Model architecture ##

# Positional encoding
# Attention layers see inputs as a set of vectors with no order. It needs way to identify snapshot order
# Add positional Encoding to the vectors so Model knows correct order