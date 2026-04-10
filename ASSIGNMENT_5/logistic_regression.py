import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report


# dataset load
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Shape:", X.shape)


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 🔥 FIXED: WITHOUT regularization
model_no_reg = LogisticRegression(penalty=None, max_iter=5000)

model_no_reg.fit(X_train, y_train)

y_pred_no_reg = model_no_reg.predict(X_test)

print("\nWITHOUT Regularization:")
print(classification_report(y_test, y_pred_no_reg))


# WITH L2 regularization
model_l2 = LogisticRegression(penalty='l2', max_iter=5000)

model_l2.fit(X_train, y_train)

y_pred_l2 = model_l2.predict(X_test)

print("\nWITH L2 Regularization:")
print(classification_report(y_test, y_pred_l2))


# ROC Curve
y_prob = model_l2.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="ROC Curve")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig("roc_curve.png")
plt.show()


# AUC Score
auc_score = roc_auc_score(y_test, y_prob)

print("\nAUC Score:", auc_score)