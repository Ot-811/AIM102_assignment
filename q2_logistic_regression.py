import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([1, 1, 0, 0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = 0.5
b = -1
lr = 0.1
iterations = 1000

losses = []

for i in range(iterations):
    z = w * X.flatten() + b
    y_hat = sigmoid(z)

    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    losses.append(loss)

    dw = np.sum((y_hat - y) * X.flatten())
    db = np.sum(y_hat - y)

    w -= lr * dw
    b -= lr * db
print(f"\nFinal Loss = {losses[-1]:.4f}")

model = LogisticRegression()
model.fit(X, y)

print("From scratch:")
print("w =", round(w, 4), "b =", round(b, 4))

print("\nSklearn:")
print("w =", round(model.coef_[0][0], 4), "b =", round(model.intercept_[0], 4))

plt.figure()
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.grid()

plt.savefig("Q2_logistic_loss.png", dpi=300, bbox_inches='tight')
plt.show()