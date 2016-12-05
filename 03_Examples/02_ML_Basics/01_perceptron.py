from helper import get_random_sample

import tensorflow as tf
import numpy as np

INPUT_SIZE = 2
DATASET_SIZE = 25
LEARNING_RATE = 0.01

# All the tensors
x = tf.placeholder(shape=(INPUT_SIZE, 1), dtype=tf.float32, name="x")
w = tf.Variable(tf.random_normal((INPUT_SIZE, 1)), name="weights")
b = tf.Variable(1.0, name="bias")
y = tf.tanh(tf.matmul(w, x, transpose_a=True) + b)
target = tf.placeholder(dtype=tf.float32, name="target")

# Defining cost function and the way how to optimize it
cost = tf.squared_difference(y, target, name="cost")
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate the calculations
dataset = []
for _ in range(DATASET_SIZE):
    sample_vector, target_value = get_random_sample()
    dataset.append(
        (np.array(sample_vector).reshape((INPUT_SIZE, 1)),
         target_value))

# Fit model and run it on the examples
init_op = tf.initialize_all_variables()
with tf.Session() as session:
    # But initialize the variables first...
    session.run(init_op)
    for epoch in range(50000):
        for sample, target_value in dataset:
            _, epoch_cost = session.run([optimizer, cost], {
                x: sample,
                target: target_value
            })

    print(session.run(y, {x: np.array([[0.1], [0.1]])}))
    print(session.run(y, {x: np.array([[0.5], [0.5]])}))
    print(session.run(y, {x: np.array([[0.7], [0.7]])}))
    print(session.run(y, {x: np.array([[0.9], [0.9]])}))
