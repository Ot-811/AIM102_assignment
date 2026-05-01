import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
y = np.array([7, 3, 1, 1, 3, 7, 13])

degrees = [2, 4, 8]
mse_list = []

x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

plt.figure()

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x)
    X_plot_poly = poly.transform(x_plot)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    mse_list.append(mse)

    y_plot = model.predict(X_plot_poly)

    plt.plot(x_plot, y_plot, label=f"Degree {d}")

plt.scatter(x, y)
plt.legend()
plt.title("Polynomial Regression Fits")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.savefig("q3_polynomial_fits.png", dpi=300, bbox_inches='tight')
plt.show()

for d, mse in zip(degrees, mse_list):
    print(f"Degree {d}: MSE = {mse:.4f}")