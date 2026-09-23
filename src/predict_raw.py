import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import kurtosis
import time


MODEL_FILE = "model.pkl"
FEATURE_DATASET = "features.csv"

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

    return features


print("==============================")
print("RAW VIBRATION PREDICTION")
print("==============================")


file_path = input(
    "\nEnter raw vibration CSV path: "
).strip()


file_path = Path(file_path)


if not file_path.exists():

    print("\nERROR: File not found.")

    exit()


model = joblib.load(MODEL_FILE)

training_data = pd.read_csv(
    FEATURE_DATASET
)

feature_columns = [
    column
    for column in training_data.columns
    if column not in ["file", "label"]
]


print("\nExtracting vibration features...")


start = time.perf_counter()

features = extract_file_features(
    file_path
)

feature_time = (
    time.perf_counter() - start
) * 1000


X = pd.DataFrame(
    [features]
)

X = X[feature_columns]


print("Feature extraction completed.")


print("\nRunning Random Forest prediction...")


start = time.perf_counter()

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

prediction_time = (
    time.perf_counter() - start
) * 1000


confidence = probabilities.max() * 100


total_time = feature_time + prediction_time


print("\n==============================")
print("PREDICTION RESULT")
print("==============================")


print("\nFile:")
print(file_path)


print("\nPredicted condition:")
print(prediction.upper())


print(f"\nConfidence: {confidence:.2f}%")


print(
    f"\nFeature extraction time: "
    f"{feature_time:.3f} ms"
)


print(
    f"Model inference time: "
    f"{prediction_time:.3f} ms"
)


print(
    f"Total processing time: "
    f"{total_time:.3f} ms"
)


if prediction == "normal":

    print("\nStatus: NORMAL")

    print(
        "Action: Fan operating normally. "
        "Continue monitoring."
    )

else:

    print("\nStatus: FAULT DETECTED")

    recommendations = {

        "imbalance":
            "Inspect rotor/fan balance and check for uneven mass distribution.",

        "horizontal-misalignment":
            "Inspect horizontal shaft alignment and coupling.",

        "vertical-misalignment":
            "Inspect vertical shaft alignment and mounting.",

        "underhang":
            "Inspect underhung rotor, bearings and support condition.",

        "overhang":
            "Inspect overhung rotor, bearings and shaft condition."
    }

    print(
        "Recommended action:",
        recommendations.get(
            prediction,
            "Inspect the ventilation fan."
        )
    )


print("\nClass probabilities:")

for class_name, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{class_name:25s}: "
        f"{probability * 100:.2f}%"
    )


print("\n==============================")
print("30 EXTRACTED FEATURES")
print("==============================")


for column in feature_columns:

    print(
        f"{column:25s}: "
        f"{X.iloc[0][column]:.6f}"
    )