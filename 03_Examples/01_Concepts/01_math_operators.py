import tensorflow as tf
import numpy as np


a = tf.constant(2.5)
b = tf.constant(5.0)

with tf.Session() as session:
    # sum
    print(session.run(a + b))
    # product
    print(session.run(a * b))
    # and much more
    print(session.run(tf.sin(a * np.pi)))
