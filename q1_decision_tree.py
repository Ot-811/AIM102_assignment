import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

n_samples = 500

X_informative = np.random.randn(n_samples, 2)
y = (X_informative[:, 0]**2 + X_informative[:, 1] > 0).astype(int)

X_noise = np.random.randn(n_samples, 3)
X = np.hstack((X_informative, X_noise))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

depths = [2, 4, 6, 10]
train_acc = []
test_acc = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))

    print(f"Depth {depth}: Train={train_acc[-1]:.3f}, Test={test_acc[-1]:.3f}")

plt.figure()
plt.plot(depths, train_acc, marker='o', label='Training Accuracy')
plt.plot(depths, test_acc, marker='o', label='Testing Accuracy')

plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Depth vs Accuracy')
plt.legend()
plt.grid()

plt.savefig("Q1_decision_tree_accuracy.png", dpi=300)
plt.show()

X_vis = X[:, :2]

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.3, random_state=42
)

from matplotlib.colors import ListedColormap

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_vis, y_train_vis)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolor='k', s=20)

    plt.title(f"Q1_Decision Boundary (Depth = {depth})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(f"decision_boundary_depth_{depth}.png", dpi=300, bbox_inches='tight')
    plt.show()