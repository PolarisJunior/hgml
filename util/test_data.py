
from mnist import MNIST
import numpy as np

def get_mnist_data():
    mndata = MNIST("./data")
    training, labels_train = map(np.array, mndata.load_training())
    training = training / 255.0
    testing, labels_test = map(np.array, mndata.load_testing())
    testing = testing / 255.0
    return training, labels_train, testing, labels_test