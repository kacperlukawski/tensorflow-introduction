import tensorflow as tf


x = tf.placeholder(tf.float32, name='input')
a = tf.constant(2.0, name='a')
b = tf.constant(1.0, name='b')
y = a * x + b

with tf.Session() as session:
    for x_val in range(10):
        print(x_val, session.run(y, {x: x_val}))
