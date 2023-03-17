# import framework
from framework import *

# load dataset from tensorflow dataset(we don't use tensorflow for training dataset)
from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist():
    a = []
    b = []
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for i in range(len(train_X)):
        a.append(train_X[i].flatten()/255)
    for i in range(len(test_X)):
        b.append(test_X[i].flatten())
    return np.array(a), train_y, np.array(b), test_y

trainx, trainy, testx, testy = load_mnist()
onehot = np.zeros((trainy.size, trainy.max() + 1))
onehot[np.arange(trainy.size), trainy] = 1

# create model with our framework
class Model(PyDahoShoxa):
    def __init__(self):
        super().__init__()
        self.l1 = Layer(784, 64)
        self.l2 = Layer(64, 16)
        self.l3 = Layer(16, 10)
        self.relu = Activations('relu')
        self.softmax = Activations('softmax')
        
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        
        return x

my_model = Model()
my_model.compile(optimizer=Optimizer(params=my_model.parameters(), learning_rate=0.001), loss='cross_entropy_loss')

my_model.fit(trainx[:100], onehot[:100], batch_size=4, epochs=10)

my_model.save('model.json')