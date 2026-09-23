import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/processed/features_v2.csv")

X = df.drop(columns=["file", "label"])
y = df["label"]


def get_group(file_path):
    parts = file_path.replace("\\", "/").split("/")

    label = parts[2]

    if label == "normal":
        return "normal"

    if label in ["underhang", "overhang"]:
        return "/".join(parts[2:5])

    return "/".join(parts[2:4])

df["group"] = df["file"].apply(get_group)


fault_groups = sorted(
    df.loc[df["label"] != "normal", "group"].unique()
)

print("Total fault configurations:", len(fault_groups))

print("\nFault configurations:")
for group in fault_groups:
    print(group)


all_predictions = []
all_actual = []

group_results = []


for i, test_group in enumerate(fault_groups, start=1):

    test_mask = df["group"] == test_group

    train_mask = df["group"] != test_group

    # Keep all normal samples in training
    # because normal is not being evaluated as a held-out fault configuration.
    train_mask = train_mask | (df["label"] == "normal")

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    group_results.append({
        "group": test_group,
        "samples": len(y_test),
        "accuracy": accuracy
    })

    all_predictions.extend(predictions)
    all_actual.extend(y_test)

    print(
        f"{i:02d}/{len(fault_groups)} "
        f"{test_group}: "
        f"{accuracy * 100:.2f}% "
        f"({len(y_test)} samples)"
    )


overall_accuracy = accuracy_score(
    all_actual,
    all_predictions
)

print("\n" + "=" * 60)
print("GROUPED VALIDATION V2 — CORRECTED")
print("=" * 60)

print(
    "\nTotal held-out fault samples:",
    len(all_actual)
)

print(
    "Overall accuracy:",
    f"{overall_accuracy * 100:.2f}%"
)

print("\nClassification report:")

print(
    classification_report(
        all_actual,
        all_predictions
    )
)

cm = confusion_matrix(
    all_actual,
    all_predictions
)

print("\nConfusion matrix:")
print(cm)

results_df = pd.DataFrame(group_results)

print("\nMean group accuracy:")
print(
    f"{results_df['accuracy'].mean() * 100:.2f}%"
)

print("\nMedian group accuracy:")
print(
    f"{results_df['accuracy'].median() * 100:.2f}%"
)

print("\nLowest-performing groups:")

print(
    results_df
    .sort_values("accuracy")
    .head(10)
    .to_string(index=False)
)

results_df.to_csv(
    "results/evaluation/grouped_validation_v2_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "grouped_validation_v2_results.csv"
)