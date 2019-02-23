""" 
    Ordinary Least Squares Regression
 """
import numpy as np
from util.data_manip import *
from regression.Regression import Regression

class OlsRegression(Regression):
    """ Solve the equation (X^TX)^-1X^Ty=w """
    def train(self, x, y):
        assert(isinstance(x, np.ndarray))
        assert(isinstance(x, np.ndarray))

        x = add_intercept_column(x)
        # we need to slightly perturb the matrix to make it invertible
        perturb(x)
        self.coefficients = np.linalg.solve(x.transpose() @ x, x.transpose() @ y)
        pass


if __name__ == "__main__":
    from mnist import MNIST
    mndata = MNIST("./data")
    training, labels_train = map(np.array, mndata.load_training())
    training = training / 255.0
    testing, labels_test = map(np.array, mndata.load_testing())
    testing = testing / 255.0
    
    model = OlsRegression()
    model.train(training, labels_train)
    # print(model)
    
    