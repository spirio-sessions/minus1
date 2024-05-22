from data_preperation import dataset_snapshot

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, MultiHeadAttention, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split

## Get dataset ##
dataset_as_snapshots = dataset_snapshot.process_dataset("../datasets/GiantMIDI-PIano/surname_checked_midis_v1.2/surname_checked_midis", 0.1, 'mozart')
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

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape,X_test.shape, y_test.shape)

## Define Model architecture ##

# Positional encoding
# Attention layers see inputs as a set of vectors with no order. It needs way to identify snapshot order
# Add positional Encoding to the vectors so Model knows correct order
class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding.numpy(),
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Define a single Decoder layer with multi-head attention and feed-forward network
def transformer_decoder_layer(units, d_model, num_heads, dropout):
    inputs = Input(shape=(None, d_model))
    look_ahead_mask = Input(shape=(1, None, None))

    attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs,
                                                                          attention_mask=look_ahead_mask)
    attention1 = Dropout(dropout)(attention1)
    attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    ffn_output = Dense(units, activation='relu')(attention1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    outputs = LayerNormalization(epsilon=1e-6)(ffn_output + attention1)

    return Model(inputs=[inputs, look_ahead_mask], outputs=outputs)

# Function to build the Transformer Decoder-Only Model
def build_decoder_only_transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, target_seq_length):
    inputs = Input(shape=(target_seq_length,))
    look_ahead_mask = Input(shape=(1, target_seq_length, target_seq_length))

    # Embedding and Positional Encoding
    embedding = Embedding(vocab_size, d_model)(inputs)
    pos_encoding = PositionalEncoding(target_seq_length, d_model)(embedding)

    outputs = pos_encoding
    for _ in range(num_layers):
        outputs = transformer_decoder_layer(units, d_model, num_heads, dropout)([outputs, look_ahead_mask])

    outputs = Dense(vocab_size, activation='softmax')(outputs)

    return Model(inputs=[inputs, look_ahead_mask], outputs=outputs)

# Helper function to create a look-ahead mask
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]


# Hyperparameters
vocab_size = 88  # For each playable note
num_layers = 4
units = 512
d_model = 128
num_heads = 8
dropout = 0.1
target_seq_length = 10


# Build the model
transformer_decoder = build_decoder_only_transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, target_seq_length)
transformer_decoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer_decoder.summary()

# Prepare the look-ahead mask for training
look_ahead_mask = create_look_ahead_mask(target_seq_length)

# Train the model
transformer_decoder.fit([X_train, look_ahead_mask], y_train, validation_data=([X_val, look_ahead_mask], y_val), batch_size=32, epochs=10)