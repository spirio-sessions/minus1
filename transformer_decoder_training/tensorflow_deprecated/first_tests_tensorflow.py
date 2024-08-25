# Just some first test in no way functional


# Load midi snapshots

snapshots = dataset_snapshot.process_dataset("../../datasets/test_files_and_configs", 0.1)

print(snapshots)

# Daten aufteilen

from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(snapshots, test_size=0.2, random_state=42)

# daten in batches aufteilen


def create_batches(data, batch_size):
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        yield batch_data


batch_size = 32
train_batches = create_batches(train_data, batch_size)
val_batches = create_batches(val_data, batch_size)

print(train_batches)
print(val_batches)

### Modellarchitektur ###

import tensorflow as tf


# eingabe/ausgabedimensionen

input_dim = 128  # Anzahl der MIDI-Noten
output_dim = input_dim  # Ausgabesequenz hat die gleiche Dimension wie Eingabe

# hyperparameter

num_layers = 4
d_model = 128  # Dimensionalität der Embeddings
num_heads = 8
dff = 512  # Dimension des Feedforward-Netzwerks
dropout_rate = 0.2

# positional encodings

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# klasse für schicht für Transformer-Decoder

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super(TransformerDecoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# klasse für gesamten transformer decoder

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, mask)
        return x


# Das modell selbst erstellen

def create_transformer_decoder_model(input_dim, output_dim, num_layers, d_model, num_heads, dff, dropout_rate):
    inputs = tf.keras.Input(shape=(None, input_dim))
    mask = tf.keras.Input(shape=(1, 1, None))  # Maske für die Padding-Positionen

    # Positionales Encoding hinzufügen
    position = 1000
    x = PositionalEncoding(position, d_model)(inputs)

    # Transformer-Decoder-Schichten hinzufügen
    x = TransformerDecoder(num_layers, d_model, num_heads, dff, dropout_rate)(x, training=True, mask=mask)

    # Ausgangsschicht
    outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[inputs, mask], outputs=outputs)


### Modell trainieren ###

model = create_transformer_decoder_model(input_dim, output_dim, num_layers, d_model, num_heads, dff, dropout_rate)

# Verlustfunktion + optimierer

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# trainingsfunktion

@tf.function
def train_step(inputs, targets, mask):
    with tf.GradientTape() as tape:
        predictions = model([inputs, mask], training=True)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Trainingsablauf

epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for batch in train_batches:
        inputs, targets = batch[:, :-1, :], batch[:, 1:, :]
        mask = tf.expand_dims(tf.cast(inputs != 0, dtype=tf.float32), axis=1)  # Maske für Padding-Positionen
        loss = train_step(inputs, targets, mask)
        total_loss += loss
    average_loss = total_loss / len(train_batches)
    print("Epoch {}: Loss {:.4f}".format(epoch + 1, average_loss))


# Modell bewerten

def evaluate_model(model, val_batches):
    total_loss = 0.0
    for batch in val_batches:
        inputs, targets = batch[:, :-1, :], batch[:, 1:, :]
        mask = tf.expand_dims(tf.cast(inputs != 0, dtype=tf.float32), axis=1)  # Maske für Padding-Positionen
        predictions = model([inputs, mask], training=False)
        loss = loss_object(targets, predictions)
        total_loss += loss
    average_loss = total_loss / len(val_batches)
    return average_loss

val_loss = evaluate_model(model, val_batches)
print("Validation Loss:", val_loss.numpy())
