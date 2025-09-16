import numpy as np
import matplotlib.pyplot as plt
from model import Dense, ReLU, MSELoss

X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = X**3 + 0.1 * np.random.randn(200, 1)

hidden = Dense(1, 10)
relu = ReLU()
output = Dense(10, 1)
loss_fn = MSELoss()

lr = 0.01
epochs = 1000
losses = []

for i in range(epochs):
    h = hidden.forward(X)
    h_relu = relu.forward(h)
    y_pred = output.forward(h_relu)
    
    loss = loss_fn.forward(y_pred, y)
    losses.append(loss)
    
    dloss = loss_fn.backward()
    doutput = output.backward(dloss, lr)
    drelu = relu.backward(doutput, lr)
    dhidden = hidden.backward(drelu, lr)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.show()

y_pred_final = output.forward(relu.forward(hidden.forward(X)))
plt.scatter(X, y, label="True")
plt.scatter(X, y_pred_final, label="Predicted")
plt.legend()
plt.title("Predicted vs True Values")
plt.show()
