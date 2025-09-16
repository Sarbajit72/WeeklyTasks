import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
    def backward(self, dZ, lr):
        dW = np.dot(self.X.T, dZ) / self.X.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]
        self.W -= lr * dW
        self.b -= lr * db
        dX = np.dot(dZ, self.W.T)
        return dX

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dZ, lr):
        return dZ * (self.X > 0)

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
