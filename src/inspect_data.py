import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/mafaulda")

files = list(DATA_DIR.rglob("*.csv"))

labels = []

for file in files:
    relative_path = file.relative_to(DATA_DIR)
    label = relative_path.parts[0]
    labels.append(label)

counts = Counter(labels)

print("Total CSV files:", len(files))

print("\nClass distribution:")
for label, count in sorted(counts.items()):
    print(f"{label:25} {count}")

print("\nTotal classes:", len(counts))