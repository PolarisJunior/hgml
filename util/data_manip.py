
import numpy as np
import operator

def add_intercept_column(matrix):
    """ Pushes a ones column onto the leftmost side
        of a 2d array i.e a matrix """
    assert(isinstance(matrix, np.ndarray))
    assert(len(matrix.shape) == 2)

    ret = np.ones(tuple(map(operator.add, matrix.shape, (0, 1))))
    ret[:,1:] = matrix
    return ret
    pass

def perturb(arr, inplace=True, sigma=0.00001):
    """ Adds small gaussian noise to an array """
    assert(isinstance(arr, np.ndarray))

    if inplace:
        arr += sigma * np.random.randn(*arr.shape)
        return arr
    else:
        return arr + sigma * np.random.randn(*arr.shape)

def one_hot_encoding(vec, dimensions=None):
    """ Converts a list of integer values to
        a one hot encoding """
    if dimensions is None:
        dimensions = np.max(vec) + 1
    
    encoded = np.zeros((len(vec), dimensions))
    encoded[np.arange(len(vec)), vec] = 1
    return encoded
    pass

def demean():
    pass

if __name__ == "__main__":
    x = add_intercept_column(np.array([[1, 2], [3, 4]]))
    assert(x.shape == (2, 3))

    ohe = one_hot_encoding(np.array([3]))
    assert(np.array_equal(ohe, np.array([[0, 0, 0, 1]])))
    pass