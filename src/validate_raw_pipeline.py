import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, classification_report


MODEL_FILE = "model.pkl"
DATA_DIR = Path("data/mafaulda")
OUTPUT_FILE = "raw_pipeline_results.csv"

VIBRATION_COLUMNS = [1, 2, 3, 4, 5, 6]


def extract_features(signal):

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    kurt = kurtosis(signal, fisher=False)
    peak_to_peak = np.ptp(signal)
    crest_factor = np.max(np.abs(signal)) / (rms + 1e-10)

    return {
        "rms": rms,
        "std": std,
        "kurtosis": kurt,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor
    }


def extract_file_features(file_path):

    start = time.perf_counter()

    data = pd.read_csv(
        file_path,
        header=None
    )

    features = {}

    for column in VIBRATION_COLUMNS:

        signal = data.iloc[
            :,
            column
        ].to_numpy(
            dtype=np.float64
        )

        channel_features = extract_features(signal)

        for feature_name, value in channel_features.items():

            features[
                f"ch{column}_{feature_name}"
            ] = value

    extraction_time = (
        time.perf_counter() - start
    ) * 1000

    return features, extraction_time


model = joblib.load(MODEL_FILE)

feature_data = pd.read_csv(
    "features.csv"
)

feature_columns = [
    column
    for column in feature_data.columns
    if column not in ["file", "label"]
]


print("==============================")
print("RAW PIPELINE VALIDATION")
print("==============================")


files = list(
    DATA_DIR.rglob("*.csv")
)


print("\nTotal raw files:", len(files))


results = []


for i, file_path in enumerate(files, start=1):

    try:

        relative_path = file_path.relative_to(
            DATA_DIR
        )

        actual_label = relative_path.parts[0]


        features, extraction_time = (
            extract_file_features(file_path)
        )


        X = pd.DataFrame(
            [features]
        )

        X = X[feature_columns]


        start = time.perf_counter()

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        inference_time = (
            time.perf_counter() - start
        ) * 1000


        confidence = probabilities.max() * 100


        total_time = (
            extraction_time +
            inference_time
        )


        results.append({

            "file": str(file_path),

            "actual": actual_label,

            "predicted": prediction,

            "correct": actual_label == prediction,

            "confidence": confidence,

            "feature_extraction_ms":
                extraction_time,

            "inference_ms":
                inference_time,

            "total_processing_ms":
                total_time
        })


        if i % 100 == 0:

            print(
                f"Processed "
                f"{i}/{len(files)} files"
            )


    except Exception as e:

        print(
            f"Error processing "
            f"{file_path}: {e}"
        )


results_df = pd.DataFrame(results)


accuracy = accuracy_score(
    results_df["actual"],
    results_df["predicted"]
)


print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")


print(
    f"\nTotal files tested: "
    f"{len(results_df)}"
)


print(
    f"Correct predictions: "
    f"{results_df['correct'].sum()}"
)


print(
    f"Incorrect predictions: "
    f"{(~results_df['correct']).sum()}"
)


print(
    f"\nRaw pipeline accuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"\nAverage feature extraction: "
    f"{results_df['feature_extraction_ms'].mean():.3f} ms"
)


print(
    f"Average model inference: "
    f"{results_df['inference_ms'].mean():.3f} ms"
)


print(
    f"Average total processing: "
    f"{results_df['total_processing_ms'].mean():.3f} ms"
)


print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")


print(
    classification_report(
        results_df["actual"],
        results_df["predicted"]
    )
)


print("\n==============================")
print("CLASS-WISE RESULTS")
print("==============================")


class_results = (
    results_df
    .groupby("actual")["correct"]
    .agg(
        total="count",
        correct="sum"
    )
)


class_results["accuracy"] = (
    class_results["correct"] /
    class_results["total"] *
    100
)


print(class_results)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nDetailed results saved to: "
    f"{OUTPUT_FILE}"
)


errors = results_df[
    results_df["correct"] == False
]


print("\n==============================")
print("MISCLASSIFIED FILES")
print("==============================")


if len(errors) == 0:

    print("\nNo misclassified files.")

else:

    print(
        errors[
            [
                "file",
                "actual",
                "predicted",
                "confidence"
            ]
        ].to_string(index=False)
    )