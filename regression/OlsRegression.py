""" 
    Ordinary Least Squares Regression
 """
import numpy as np
from util.data_manip import *
from util.test_data import get_mnist_data
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
    from evaluate.metrics import *
    training, labels_train, testing, labels_test = get_mnist_data()
    labels_train = one_hot_encoding(labels_train)
    labels_test = one_hot_encoding(labels_test)
    
    model = OlsRegression()
    model.train(training, labels_train)
    
    predictions = model.predict(training)
    print("accuracy on training %s" % get_accuracy_one_hot(predictions, labels_train))

    predictions = model.predict(testing)
    print("accuracy on testing %s" % get_accuracy_one_hot(predictions, labels_test))
    

    
    