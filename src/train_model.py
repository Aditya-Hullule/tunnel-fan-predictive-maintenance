import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET = "features.csv"
MODEL_FILE = "model.pkl"


df = pd.read_csv(DATASET)

X = df.drop(columns=["file", "label"])
y = df["label"]


print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])

print("\nClasses:")
print(y.value_counts())


print("\n==============================")
print("DATASET CHECK")
print("==============================")

print("Total rows:", len(df))
print("Unique files:", df["file"].nunique())

if df["file"].nunique() == len(df):
    print("Each row has a unique file.")
    print("Random train/test split is appropriate.")
else:
    print("WARNING: Multiple rows belong to the same file.")
    print("Random splitting may cause data leakage.")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Training completed.")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")


feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)


print("\nTop 15 Important Features:")

print(
    feature_importance.head(15).to_string(index=False)
)


feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="importance",
    y="feature"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Features")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("\n==============================")
print("5-FOLD CROSS-VALIDATION")
print("==============================")

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

print("\nFold accuracies:")

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nMean accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard deviation: {cv_scores.std() * 100:.2f}%")


joblib.dump(model, MODEL_FILE)


print("\n==============================")
print("FILES SAVED")
print("==============================")

print("Model saved as:", MODEL_FILE)
print("Confusion matrix saved as: confusion_matrix.png")
print("Feature importance saved as: feature_importance.png")
print("Feature importance data saved as: feature_importance.csv")


joblib.dump(model, MODEL_FILE)


print("\n==============================")
print("FILES SAVED")
print("==============================")

print("Model saved as:", MODEL_FILE)
print("Confusion matrix saved as: confusion_matrix.png")
print("Feature importance saved as: feature_importance.png")
print("Feature importance data saved as: feature_importance.csv")