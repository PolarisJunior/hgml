
import numpy as np
from regression.Regression import Regression
from util.data_manip import add_intercept_column, perturb

class RidgeRegression(Regression):
    def __init__(self, lamb = 0.0001):
        super().__init__()
        self.lamb = lamb

    def train(self, x, y):
        """ Solve the equation (X^TX+lambda*I)^-1X^Ty=w """
        assert(isinstance(x, np.ndarray))
        assert(isinstance(x, np.ndarray))

        x = add_intercept_column(x)
        self.coefficients = np.linalg.solve(x.transpose() @ x + np.identity(x.shape[1]) * self.lamb, x.transpose() @ y)


if __name__ == "__main__":
    from evaluate.eval_regression import evaluate_model
    model = RidgeRegression()
    evaluate_model(model)

