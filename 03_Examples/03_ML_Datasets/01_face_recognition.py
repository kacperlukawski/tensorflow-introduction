# This example tries to create a model able to handle face recognition.
# The dataset is taken from "CMU Face Images":
# http://kdd.ics.uci.edu/databases/faces/faces.html
# It contains at least 28 different images for 20 people. Created model
# should be able to recognize the name of the person.
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


# Helper functions
def read_pgm_file(file_path):
    """
    Opens file with given path and convert it to numpy matrix
    of pixels. It expects to have a valid PGM file.
    :param file_path: path to PGM file to be read
    :return: numpy array with pixel values
    """
    with open(file_path, 'r') as fh:
        lines = fh.readlines()
        return np.array([line.split() for line in lines[3:]], dtype=np.int8)
    return np.ones((IMAGE_WIDTH, IMAGE_HEIGHT))


def prepare_samples(dataset):
    """
    Prepares iterable of input vectors from the dataset files,
    as well as another iterable with the targets.
    :param dataset: dataset to be converted
    :return:
    """
    samples, targets = [], []
    for entry in dataset:
        class_idx, _, file_path = entry
        # Prepare sample
        sample = read_pgm_file(file_path).astype(np.float32).reshape(IMAGE_WIDTH * IMAGE_HEIGHT)
        samples.append(sample)
        # Prepare target
        target = np.zeros((CLASSES_COUNT, ))
        target[class_idx] = 1
        targets.append(target)
    return samples, targets


# Read all the file names and form the dataset
class_idx, dataset = 0, list()
for directory in glob.glob('./faces/*'):
    person = os.path.basename(directory)
    for filename in glob.glob(directory + '/*[!0-9].pgm'):
        dataset.append((class_idx, person, filename))
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
    layer_name = 'layer_%i' % (i,)
    weights_name = 'weights_%i' % (i,)
    biases_name = 'biases_%i' % (i,)
    # create weights and biases
    last_layer_shape = last_layer.get_shape()
    weights = tf.Variable(tf.random_normal(shape=(int(last_layer_shape[1]), HIDDEN_LAYERS_SIZE[i])),
                          name=weights_name)
    biases = tf.Variable(tf.constant(INITIAL_BIAS, shape=(1, HIDDEN_LAYERS_SIZE[i])),
                         name=biases_name)
    # Create a new hidden layer and set it as a new last one
    last_layer = tf.sigmoid(tf.matmul(last_layer, weights) + biases, name=layer_name)

# Connect the output layer and create whole NN
last_layer_shape = last_layer.get_shape()
weights = tf.Variable(tf.random_normal(shape=(int(last_layer_shape[1]), CLASSES_COUNT)),
                      name='weights_output')
biases = tf.Variable(tf.constant(INITIAL_BIAS, shape=(1, CLASSES_COUNT)), name='biases_output')
output_vector = tf.add(tf.matmul(last_layer, weights), biases, name='output_vector')

# Create cost function of the created network and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_vector, target_vector))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Create accuracy calculation
correct_prediction = tf.equal(tf.arg_max(output_vector, 1), tf.arg_max(target_vector, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run training
init_op = tf.initialize_all_variables()
with tf.Session() as session:
    # Initialize all variables
    session.run(init_op)

    # Train the model
    samples, targets = prepare_samples(train_set)
    for epoch in range(EPOCHS):
        _, epoch_cost = session.run([optimizer, cost], {
            input_vector: samples,
            target_vector: targets
        })
        print('Epoch', epoch, 'cost:', epoch_cost)

    # Test created model
    samples, targets = prepare_samples(test_set)
    prediction_accuracy = session.run(accuracy, {
        input_vector: samples,
        target_vector: targets
    })

    # Check correct predictions
    print(prediction_accuracy)
