import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

DATA_DIR = Path("data/mafaulda")
OUTPUT_FILE = "features.csv"

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


rows = []

files = list(DATA_DIR.rglob("*.csv"))

print("Total files:", len(files))

for i, file in enumerate(files, start=1):

    try:
        data = pd.read_csv(file, header=None)

        row = {
            "file": str(file)
        }

        relative_path = file.relative_to(DATA_DIR)
        row["label"] = relative_path.parts[0]

        for column in VIBRATION_COLUMNS:

            signal = data.iloc[:, column].to_numpy(dtype=np.float64)

            features = extract_features(signal)

            for feature_name, value in features.items():
                row[f"ch{column}_{feature_name}"] = value

        rows.append(row)

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)} files")

    except Exception as e:
        print(f"Error processing {file}: {e}")


df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed.")
print("Dataset shape:", df.shape)
print("Saved to:", OUTPUT_FILE)

print("\nClass distribution:")
print(df["label"].value_counts())