# This example tries to create a model able to handle face recognition.
# The dataset is taken from "CMU Face Images":
# http://kdd.ics.uci.edu/databases/faces/faces.html
# It contains at least 28 different images for 20 people. Created model
# should be able to recognize the name of the person.
from helper import prepare_samples

import tensorflow as tf
import numpy as np
import random
import glob
import os

# Configuration
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 120
TRAIN_SET_FRACTION = .85
CLASSES_COUNT = 20
HIDDEN_LAYERS_SIZE = [250, ]  # [125, 35]  # 50, 35, ]  # best: [250, ]
INITIAL_BIAS = .2
LEARNING_RATE = .1
EPOCHS = 10000
ACTIVATION_FUNCTION = tf.sigmoid

# Read all the file names and form the dataset
classes, class_idx, dataset = dict(), 0, list()
for directory in glob.glob('./faces/*'):
    person = os.path.basename(directory)
    for filename in glob.glob(directory + '/*'):
        dataset.append((class_idx, person, filename))
    classes[class_idx] = person
    class_idx += 1

# Shuffle the dataset and split it into train and test sets
random.shuffle(dataset)
train_set_size = int(len(dataset) * TRAIN_SET_FRACTION)
train_set, test_set = dataset[:train_set_size], dataset[train_set_size:]

# Create the input layer. As we use the images of size 128x120, a placeholder
# will assume to have a vector of a size: IMAGE_WIDTH * IMAGE_HEIGHT
input_vector = tf.placeholder(tf.float32, shape=(None, IMAGE_WIDTH * IMAGE_HEIGHT))
target_vector = tf.placeholder(dtype=tf.float32, shape=(None, CLASSES_COUNT),
                               name='target_vector')

# Create hidden layers
last_layer = input_vector
for i in range(len(HIDDEN_LAYERS_SIZE)):
    # Create weights and biases
    last_layer_shape = last_layer.get_shape()
    weights = tf.Variable(tf.random_normal(shape=(int(last_layer_shape[1]), HIDDEN_LAYERS_SIZE[i])),
                          name='weights_%i' % (i,))
    biases = tf.Variable(tf.constant(INITIAL_BIAS, shape=(1, HIDDEN_LAYERS_SIZE[i])),
                         name='biases_%i' % (i,))
    # Create a new hidden layer and set it as a new last one
    last_layer = ACTIVATION_FUNCTION(tf.matmul(last_layer, weights) + biases, name='layer_%i' % (i,))

# Connect the output layer and create whole NN
last_layer_shape = last_layer.get_shape()
weights = tf.Variable(tf.random_normal(shape=(int(last_layer_shape[1]), CLASSES_COUNT)),
                      name='weights_output')
biases = tf.Variable(tf.constant(INITIAL_BIAS, shape=(1, CLASSES_COUNT)), name='biases_output')
output_vector = tf.add(tf.matmul(last_layer, weights), biases, name='output_vector')

# Create cost function of the created network and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_vector, target_vector))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)  # GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Create accuracy calculation
correct_prediction = tf.equal(tf.arg_max(output_vector, 1),
                              tf.arg_max(target_vector, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run training
init_op = tf.initialize_all_variables()
with tf.Session() as session:
    # Initialize all variables
    session.run(init_op)

    # Train the model
    try:
        samples, targets = prepare_samples(train_set, IMAGE_WIDTH, IMAGE_HEIGHT,
                                           CLASSES_COUNT)
        for epoch in range(EPOCHS):
            _, epoch_cost = session.run([optimizer, cost], {
                input_vector: samples,
                target_vector: targets
            })
            print('Epoch', epoch, 'cost:', epoch_cost)
    except KeyboardInterrupt as e:
        print("Training phase interrupted")

    # Test created model
    samples, targets = prepare_samples(test_set, IMAGE_WIDTH, IMAGE_HEIGHT,
                                       CLASSES_COUNT)
    prediction_accuracy = session.run(accuracy, {
        input_vector: samples,
        target_vector: targets
    })

    # Check the output for each tested file
    for entry, sample, target in zip(test_set, samples, targets):
        target_class_idx = np.argmax(target)
        predicted_class_idx = np.argmax(session.run(output_vector, {
            input_vector: (sample,)
        }))
        if target_class_idx == predicted_class_idx:
            continue
        print(entry[2],
              "wanted:", classes[target_class_idx],
              "actual:", classes[predicted_class_idx])

    # Check correct predictions
    print('Accuracy:', prediction_accuracy)
