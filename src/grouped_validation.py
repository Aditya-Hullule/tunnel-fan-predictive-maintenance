import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET = "features.csv"


df = pd.read_csv(DATASET)


def get_group(file_path):

    path = Path(file_path)

    parts = path.parts

    data_index = parts.index("data")

    relative_parts = parts[data_index + 2:-1]

    if len(relative_parts) == 0:
        return "normal"

    return "/".join(relative_parts)


df["group"] = df["file"].apply(get_group)


feature_columns = [
    column
    for column in df.columns
    if column not in ["file", "label", "group"]
]


fault_groups = df[
    df["label"] != "normal"
]["group"].unique()


all_predictions = []

group_results = []


print("==============================")
print("GROUPED VALIDATION")
print("==============================")

print(
    f"\nFault configurations tested: "
    f"{len(fault_groups)}"
)

print(
    "\nNormal samples are kept in every training set."
)


for test_group in fault_groups:

    test_mask = df["group"] == test_group

    train_mask = ~test_mask


    train_df = df[train_mask]

    test_df = df[test_mask]


    X_train = train_df[
        feature_columns
    ]

    y_train = train_df["label"]


    X_test = test_df[
        feature_columns
    ]

    y_test = test_df["label"]


    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    correct = (
        predictions == y_test.to_numpy()
    )


    accuracy = correct.mean() * 100


    group_results.append({

        "group": test_group,

        "actual_class":
            test_df["label"].iloc[0],

        "samples":
            len(test_df),

        "correct":
            correct.sum(),

        "accuracy":
            accuracy
    })


    for actual, predicted in zip(
        y_test,
        predictions
    ):

        all_predictions.append({

            "group": test_group,

            "actual": actual,

            "predicted": predicted
        })


    print(
        f"{test_group:45s} "
        f"{len(test_df):3d} samples   "
        f"Accuracy: {accuracy:6.2f}%"
    )


results_df = pd.DataFrame(
    group_results
)


prediction_df = pd.DataFrame(
    all_predictions
)


y_true = prediction_df["actual"]

y_pred = prediction_df["predicted"]


overall_accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n==============================")
print("OVERALL GROUPED RESULTS")
print("==============================")


print(
    f"\nFault configurations tested: "
    f"{len(results_df)}"
)


print(
    f"Total held-out fault samples: "
    f"{len(prediction_df)}"
)


print(
    f"\nOverall grouped accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_true,
        y_pred,
        labels=[
            "horizontal-misalignment",
            "imbalance",
            "normal",
            "overhang",
            "underhang",
            "vertical-misalignment"
        ],
        zero_division=0
    )
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")


labels = [
    "horizontal-misalignment",
    "imbalance",
    "normal",
    "overhang",
    "underhang",
    "vertical-misalignment"
]


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)


cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)


print(cm_df)


print("\n==============================")
print("GROUP ACCURACY SUMMARY")
print("==============================")


print(
    f"Mean group accuracy: "
    f"{results_df['accuracy'].mean():.2f}%"
)


print(
    f"Median group accuracy: "
    f"{results_df['accuracy'].median():.2f}%"
)


print(
    f"Minimum group accuracy: "
    f"{results_df['accuracy'].min():.2f}%"
)


print(
    f"Maximum group accuracy: "
    f"{results_df['accuracy'].max():.2f}%"
)


results_df.to_csv(
    "grouped_validation_results.csv",
    index=False
)


prediction_df.to_csv(
    "grouped_predictions.csv",
    index=False
)


print("\n==============================")
print("LOWEST-PERFORMING GROUPS")
print("==============================")


print(
    results_df
    .sort_values("accuracy")
    .head(10)
    .to_string(index=False)
)


print(
    "\nResults saved:"
)

print(
    "grouped_validation_results.csv"
)

print(
    "grouped_predictions.csv"
)