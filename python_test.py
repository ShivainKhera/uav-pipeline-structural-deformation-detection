import tensorflow as tf

print("TensorFlow version:", tf.__version__)

a = tf.constant(2)
b = tf.constant(3)
print("a + b =", (a + b).numpy())