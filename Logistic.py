"""Implement Logisitc Regression using numpy for practice"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.api import add_constant

np.random.seed(0)

class MyLogisticRegression(object):
    """Logisitc Regression Class"""
    def __init__(self, num_iterations=10000, alpha=.1):
        """Uder can specify how many iterations in GD to go through as well as the learning rate"""
        self.num_iterations = num_iterations
        self.alpha = alpha

        #Parameters
        self.W = None
        self.b = None
        self.X = None
        self.y = None

    def _initialize_parameters(self):
        """Radomly initialize params"""
        self.W = np.random.random((self.X.shape[1], 1))
        self.b = 1

    def fit(self, X, y):
        """method used to fit the coefs for the given X and y"""
        self.X = X
        self.y = y

        self._initialize_parameters()

        for _ in range(self.num_iterations):
            #calc gradient
            self._calc_gradient()

    def calculate_cost(self, X):
        """Calc loss using log loss"""
        m = self.X.shape[0]
        cost = -1/m * np.sum(self.y*np.log(self.predict(X)) + (1-self.y)*np.log(1-self.predict(X)))
        return cost

    def predict(self, X):
        """Predict unseen data"""
        Z = X.dot(self.W) + self.b
        A = 1/(1 + np.exp(-Z))
        return A

    def _calc_gradient(self):
        """Internal method to fix the coefs using GD/MLE"""
        m = self.X.shape[0]

        Z = self.X.dot(self.W) + self.b
        A = sigmoid(Z)
        #derivative all in one
        dW = 1/m * self.X.T.dot(A - self.y)
        db = np.mean(A - self.y)

        #Break out by the chain rule
        # d_sigma = -((self.y/A) - (1-self.y)/(1-A))
        # dZ = A * (1-A)
        # dW = 1/m * self.X.T.dot(d_sigma * dZ)

        self.W -= self.alpha * dW
        self.b -= self.alpha * db

def sigmoid(z):
    """sigmoid helper function"""
    return 1/(1 + np.exp(-z))

def main():
    """
        Method to test the implementation
    """
    my_log = MyLogisticRegression()
    sk_log = LogisticRegression(C=1000)

    X = np.random.random((50, 4))
    y = np.random.randint(2, size=50)[:, None]

    my_log.fit(X, y)
    sk_log.fit(X, y)

    exog = add_constant(X)
    lr = Logit(y, exog)
    lrf = lr.fit()

    # print(sk_log.coef_, my_log.W, lrf.summary())
    assert np.allclose(sk_log.coef_, my_log.W.T, .1), 'incorrect coefs'

if __name__ == '__main__':
    main()
