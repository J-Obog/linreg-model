import numpy as np 

class LinReg_Model:
    def __init__(self, learning_rate=0.01):
        self.__weigths = None
        self.__learning_rate = learning_rate

    #initialize weights
    def __init_weights(self, train_dim):
        _, n = train_dim
        self.__weigths = np.ones(n)

    #tweaking weights with gradient descent
    def train(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.__init_weights(X.shape)
        print(self.__weigths)

    #test model with inputs
    def predict(self, X):
        return np.array([np.dot(self.__weigths, x) for x in X])
    
    
