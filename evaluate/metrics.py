
import numpy as np

def get_accuracy_one_hot(predictions, actuals):
    correct = np.argmax(predictions, axis=1) == np.argmax(actuals, axis=1)
    correct = np.sum(correct)
    return correct / len(predictions)
    pass