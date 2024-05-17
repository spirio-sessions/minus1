import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

print(tf.reduce_sum(tf.random.normal([1000, 1000])))