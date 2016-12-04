import tensorflow as tf


x = tf.Variable(0.0, name="x")
a = tf.constant(2.0, name="a")
b = tf.constant(1.0, name="b")
y = a * x + b

with tf.Session() as session:
    for x_val in range(10):
        op = x.assign(x_val)
        session.run(op)
        # calculate the value of y
        print(x_val, session.run(y))
