import numpy as np


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


def prepare_samples(dataset, image_width, image_height, classes_count):
    """
    Prepares iterable of input vectors from the dataset files,
    as well as another iterable with the targets.
    :param dataset: dataset to be converted
    :param image_width: input image width
    :param image_height: input image height
    :param classes_count: numer of output classes
    :return: two iterables: samples and their targets
    """
    samples, targets = [], []
    for entry in dataset:
        class_idx, _, file_path = entry
        # Prepare sample
        sample = read_pgm_file(file_path).astype(np.float32)\
            .reshape(image_width * image_height)
        samples.append(sample)
        # Prepare target
        target = np.zeros((classes_count, ))
        target[class_idx] = 1
        targets.append(target)
    return samples, targets
