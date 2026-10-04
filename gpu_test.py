import tensorflow as tf

print("✅ TensorFlow version:", tf.__version__)

# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("🚀 GPU detected:", gpus)
else:
    print("⚠️ No GPU detected, running on CPU")

# Simple computation to check GPU usage
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:\n", c.numpy())
