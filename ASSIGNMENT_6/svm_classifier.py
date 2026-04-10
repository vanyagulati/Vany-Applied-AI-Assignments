import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Original Shape:", X.shape)


# PCA reduce to 2 components (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Reduced Shape:", X_pca.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)


# Train SVM with different kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy using {kernel} kernel:", acc)


# GridSearchCV for best parameters
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# Decision boundary plot
model = grid.best_estimator_

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 1),
    np.arange(y_min, y_max, 1)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')

plt.title("SVM Decision Boundary (RBF Kernel)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("svm_decision_boundary.png")
plt.show()