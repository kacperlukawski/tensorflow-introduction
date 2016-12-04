import random


def get_random_sample():
    """
    Creates a random sample for the purposes of perceptron example.
    This example expects to belong to class "1" if it is above the
    line created by a function: f(x) = -x + 1.5 and to "0" if it lies
    below or on it. Each value of created vector is chosen from
    the [0, 1) interval.
    :return:
    """
    x, y = random.random(), random.random()
    return (x, y), int(y > -x + 1.5)
