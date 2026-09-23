import pandas as pd
import joblib
import time


MODEL_FILE = "model.pkl"
DATASET = "features.csv"


model = joblib.load(MODEL_FILE)

df = pd.read_csv(DATASET)

feature_columns = [
    col for col in df.columns
    if col not in ["file", "label"]
]


print("==============================")
print("TUNNEL FAN PREDICTION SYSTEM")
print("==============================")


sample_number = int(
    input(f"\nEnter sample number (0-{len(df)-1}): ")
)


if sample_number < 0 or sample_number >= len(df):
    print("Invalid sample number.")
    exit()


sample = df.iloc[[sample_number]]

X_sample = sample[feature_columns]


start = time.perf_counter()

prediction = model.predict(X_sample)[0]

probabilities = model.predict_proba(X_sample)[0]

inference_time = (
    time.perf_counter() - start
) * 1000


confidence = probabilities.max() * 100

actual = sample["label"].iloc[0]

filename = sample["file"].iloc[0]


print("\n==============================")
print("PREDICTION RESULT")
print("==============================")

print("\nFile:")
print(filename)

print("\nActual condition:")
print(actual.upper())

print("\nPredicted condition:")
print(prediction.upper())

print(f"\nConfidence: {confidence:.2f}%")

print(f"Inference time: {inference_time:.3f} ms")


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