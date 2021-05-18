import numpy as np 

class LinReg_Model:
    def __init__(self, iterations=1000, learning_rate=0.01):
        self.__weigths = None
        self.__learning_rate = learning_rate
        self.__iterations = iterations

    def grad_descent(self):
        for itr in self.__iterations:
            pass
        return 0

    #tweaking weights with gradient descent
    def train(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        m,n = X.shape
        self.__weigths = np.ones(n + 1)
        B = np.ones((m, 1)) #bias matrix
        X = np.hstack((X, B))

    #test model with inputs
    def predict(self, X):
        return np.array([np.dot(self.__weigths, x) for x in X])
    
    
