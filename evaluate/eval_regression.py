
from evaluate.metrics import *
from util.test_data import get_mnist_data
from util.data_manip import one_hot_encoding

def evaluate_model(model):
    training, labels_train, testing, labels_test = get_mnist_data()
    labels_train = one_hot_encoding(labels_train)
    labels_test = one_hot_encoding(labels_test)

    model.train(training, labels_train)
    
    predictions = model.predict(training)
    print("accuracy on training %s" % get_accuracy_one_hot(predictions, labels_train))

    predictions = model.predict(testing)
    print("accuracy on testing %s" % get_accuracy_one_hot(predictions, labels_test))
    pass