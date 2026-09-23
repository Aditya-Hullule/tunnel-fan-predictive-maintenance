import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/features_v2.csv")

X = df.drop(columns=["file", "label"])
y = df["label"]

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\n5-Fold Cross Validation:")
print("Fold scores:", cv_scores)
print("Mean accuracy:", f"{cv_scores.mean() * 100:.2f}%")
print("Standard deviation:", f"{cv_scores.std() * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("V2 Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig("results/evaluation/confusion_matrix_v2.png", dpi=300)
plt.close()

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    "results/features/feature_importance_v2.csv",
    index=False
)

print("\nTop 20 Features:")

print(
    feature_importance.head(20).to_string(index=False)
)

joblib.dump(model, "models/model_v2.pkl")

print("\nModel saved to: models/model_v2.pkl")

print("Confusion matrix saved to: results/evaluation/confusion_matrix_v2.png")

print("Feature importance saved to: results/features/feature_importance_v2.csv")