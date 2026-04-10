import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report


# dataset load
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Loaded Successfully")


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# models
lr = LogisticRegression(max_iter=5000)
svm = SVC(probability=True)
rf = RandomForestClassifier(random_state=42)


models = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf
}


# evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))


# ROC curve plotting
plt.figure()

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")


plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("roc_comparison.png")
plt.show()


# 5-fold cross validation
print("\nCross Validation Scores:")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f}")