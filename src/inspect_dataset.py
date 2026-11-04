
import os
import sys
import pickle
import pandas as pd
import numpy as np


# Step 1: Custom unpickler for older Pandas files
class CompatibleUnpickler(pickle.Unpickler):

    def find_class(self, module, name):

        # Fix old Pandas module paths
        if module.startswith("pandas.indexes"):
            module = module.replace(
                "pandas.indexes",
                "pandas.core.indexes",
                1
            )

        return super().find_class(module, name)


# Step 2: Dataset path
DATASET_PATH = "data/raw/LSWMD.pkl"

print("Loading wafer dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("LSWMD.pkl not found!")


# Step 3: Load using compatibility unpickler
with open(DATASET_PATH, "rb") as file:
    df = CompatibleUnpickler(
        file,
        encoding="latin1"
    ).load()


print("\n========== DATASET OVERVIEW ==========")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.drop(columns=["waferMap"]).head())

print("\nColumn data types:")
print(df.dtypes)


# Step 4: Inspect first wafer
first_wafer = df.iloc[0]["waferMap"]

print("\nFirst wafer map:")
print("Shape:", first_wafer.shape)
print("Unique values:", np.unique(first_wafer))


# Step 5: Defect distribution
def clean_label(value):

    if value is None:
        return "Unknown"

    while isinstance(value, (list, tuple, np.ndarray)):

        if len(value) == 0:
            return "Unknown"

        value = value[0]

    return str(value)


labels = df["failureType"].apply(clean_label)

print("\n========== DEFECT DISTRIBUTION ==========")

print(labels.value_counts(dropna=False))

print("\nDataset inspection completed!")


import matplotlib.pyplot as plt

# Extract labels from nested arrays
def get_label(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return "Unknown"
        value = value[0]

    if value is None or pd.isna(value):
        return "Unknown"

    return str(value)


df["cleanLabel"] = df["failureType"].apply(get_label)

# Categories we want to visualise
categories = [
    "none",
    "Center",
    "Donut",
    "Edge-Ring",
    "Edge-Loc",
    "Loc",
    "Random",
    "Scratch",
    "Near-full"
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, category in zip(axes.flat, categories):

    # Find the first wafer belonging to this category
    matching = df[df["cleanLabel"] == category]

    if len(matching) == 0:
        ax.set_title(f"{category} - Not found")
        ax.axis("off")
        continue

    wafer = matching.iloc[0]["waferMap"]

    ax.imshow(wafer, cmap="viridis", interpolation="nearest")
    ax.set_title(category)
    ax.axis("off")

plt.suptitle("WM-811K Wafer Defect Patterns", fontsize=16)
plt.tight_layout()

import os
os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/wafer_examples.png", dpi=150)
print("\nVisualisation saved to outputs/wafer_examples.png")

plt.show()

