""" 
    Ordinary Least Squares Regression
 """
import numpy as np
from util.data_manip import *
from regression.Regression import Regression

class OlsRegression(Regression):
    def train(self, x, y):
        """ Solve the equation (X^TX)^-1X^Ty=w """
        assert(isinstance(x, np.ndarray))
        assert(isinstance(x, np.ndarray))

        x = add_intercept_column(x)
        # we need to slightly perturb the matrix to make it invertible
        perturb(x)
        self.coefficients = np.linalg.solve(x.transpose() @ x, x.transpose() @ y)
        pass

if __name__ == "__main__":
    from evaluate.eval_regression import evaluate_model
    model = OlsRegression()
    evaluate_model(model)

    

    
    