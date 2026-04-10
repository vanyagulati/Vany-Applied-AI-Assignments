import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve


# dataset load
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Shape:", X.shape)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Random Forest model with OOB score
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

print("OOB Score:", rf.oob_score_)


# feature importance plot
importances = rf.feature_importances_

plt.bar(range(len(importances)), importances)

plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")

plt.savefig("feature_importance.png")
plt.show()


# compare different n_estimators
estimators = [10, 50, 100, 200]
scores = []

for n in estimators:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))


plt.plot(estimators, scores, marker='o')

plt.title("Performance vs n_estimators")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")

plt.savefig("estimators_comparison.png")
plt.show()


# learning curve
train_sizes, train_scores, test_scores = learning_curve(
    rf, X, y, cv=5
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Score")

plt.legend()

plt.savefig("learning_curve.png")
plt.show()