import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


MODEL_FILE = "model.pkl"
DATASET = "features.csv"


df = pd.read_csv(DATASET)

X = df.drop(columns=["file", "label"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = joblib.load(MODEL_FILE)


print("==============================")
print("MODEL INFERENCE TEST")
print("==============================")


start = time.perf_counter()

predictions = model.predict(X_test)

total_time = time.perf_counter() - start


accuracy = accuracy_score(
    y_test,
    predictions
)


average_time = (
    total_time / len(X_test)
) * 1000


print(f"\nTest samples: {len(X_test)}")

print(f"Accuracy: {accuracy * 100:.2f}%")

print(
    f"Total inference time: "
    f"{total_time * 1000:.2f} ms"
)

print(
    f"Average inference time/sample: "
    f"{average_time:.4f} ms"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


results = df.loc[
    X_test.index,
    ["file", "label"]
].copy()

results["predicted"] = predictions

results["correct"] = (
    results["label"] ==
    results["predicted"]
)


results.to_csv(
    "prediction_results.csv",
    index=False
)


errors = results[
    results["correct"] == False
]


print("\n==============================")
print("MISCLASSIFICATIONS")
print("==============================")

print(
    f"Incorrect predictions: "
    f"{len(errors)}"
)

print(
    f"Correct predictions: "
    f"{len(results) - len(errors)}"
)


print("\nResults saved as: prediction_results.csv")