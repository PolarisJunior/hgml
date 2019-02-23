""" base class for Regression models """

import numpy as np
from util.data_manip import *

class Regression:
    def __init__(self):
        pass

    def predict(self, x):
        assert(isinstance(x, np.ndarray))
        if self.coefficients is None:
            return None
        else:
            return add_intercept_column(x) @ self.coefficients
        pass

    def __str__(self):
        return ("coef: %s" % (self.coefficients))
