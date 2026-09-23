import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

DATA_DIR = Path("data/raw/mafaulda")
OUTPUT_FILE = "data/processed/features_v2.csv"

VIBRATION_COLUMNS = [1, 2, 3, 4, 5, 6]


def extract_time_features(signal):
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


def extract_frequency_features(signal):
    signal = signal - np.mean(signal)

    n = len(signal)

    fft_values = np.fft.rfft(signal)
    magnitude = np.abs(fft_values) / n

    frequencies = np.fft.rfftfreq(n, d=1.0)

    if len(magnitude) > 1:
        magnitude[0] = 0

    total_magnitude = np.sum(magnitude) + 1e-10

    dominant_index = np.argmax(magnitude)

    dominant_frequency = frequencies[dominant_index]
    dominant_amplitude = magnitude[dominant_index]

    spectral_centroid = np.sum(
        frequencies * magnitude
    ) / total_magnitude

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2) * magnitude
        ) / total_magnitude
    )

    spectral_energy = np.sum(magnitude ** 2)

    return {
        "dominant_frequency": dominant_frequency,
        "dominant_amplitude": dominant_amplitude,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_energy": spectral_energy
    }


rows = []

files = list(DATA_DIR.rglob("*.csv"))

print("Total files:", len(files))

for i, file in enumerate(files, start=1):

    try:
        data = pd.read_csv(file, header=None)

        row = {"file": str(file)}

        relative_path = file.relative_to(DATA_DIR)

        row["label"] = relative_path.parts[0]

        for column in VIBRATION_COLUMNS:

            signal = data.iloc[:, column].to_numpy(dtype=np.float64)

            time_features = extract_time_features(signal)

            for feature_name, value in time_features.items():
                row[f"ch{column}_{feature_name}"] = value

            frequency_features = extract_frequency_features(signal)

            for feature_name, value in frequency_features.items():
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
print("Number of features:", len(df.columns) - 2)
print("Saved to:", OUTPUT_FILE)

print("\nClass distribution:")
print(df["label"].value_counts())
