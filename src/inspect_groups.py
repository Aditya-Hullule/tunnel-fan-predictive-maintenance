import pandas as pd
from pathlib import Path


DATASET = "features.csv"

df = pd.read_csv(DATASET)


def get_group(file_path):

    path = Path(file_path)

    parts = path.parts

    data_index = parts.index("data")

    relative_parts = parts[data_index + 2:-1]

    if len(relative_parts) == 0:
        return "default"

    return "/".join(relative_parts)


df["group"] = df["file"].apply(get_group)


print("==============================")
print("FAULT CONFIGURATION GROUPS")
print("==============================")


groups = (
    df.groupby(["label", "group"])
    .size()
    .reset_index(name="samples")
)


for label in df["label"].unique():

    print(f"\n{label}")
    print("-" * len(label))

    class_groups = groups[
        groups["label"] == label
    ]

    for _, row in class_groups.iterrows():

        print(
            f"{row['group']:30s} "
            f"{row['samples']} samples"
        )


print("\n==============================")
print("TOTAL GROUPS")
print("==============================")

print(
    groups.groupby("label").size()
)
