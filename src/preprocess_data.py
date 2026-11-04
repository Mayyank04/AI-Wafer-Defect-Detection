
import os
import pickle
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# 1. Compatibility with the older WM-811K pickle
class CompatibleUnpickler(pickle.Unpickler):

    def find_class(self, module, name):

        if module.startswith("pandas.indexes"):
            module = module.replace(
                "pandas.indexes",
                "pandas.core.indexes",
                1
            )

        return super().find_class(module, name)


# 2. Configuration
DATASET_PATH = "data/raw/LSWMD.pkl"
OUTPUT_PATH = "data/processed/wafer_dataset_64.npz"

IMAGE_SIZE = 64
MAX_PER_CLASS = 1000

CLASS_NAMES = [
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

print("Starting preprocessing...")


# 3. Load dataset
with open(DATASET_PATH, "rb") as file:

    df = CompatibleUnpickler(
        file,
        encoding="latin1"
    ).load()

print("Original dataset:", df.shape)


# 4. Extract readable labels
def clean_label(value):

    while isinstance(value, (list, tuple, np.ndarray)):

        if len(value) == 0:
            return "Unknown"

        value = value[0]

    if value is None:
        return "Unknown"

    return str(value)


df["cleanLabel"] = df["failureType"].apply(clean_label)


# 5. Keep only known classes
df = df[df["cleanLabel"].isin(CLASS_NAMES)].copy()

print("Labelled samples:", len(df))


# 6. Create a balanced experimental dataset
balanced_parts = []

for category in CLASS_NAMES:

    category_df = df[df["cleanLabel"] == category]

    sample_count = min(
        MAX_PER_CLASS,
        len(category_df)
    )

    selected = category_df.sample(
        n=sample_count,
        random_state=42
    )

    balanced_parts.append(selected)

    print(category, ":", sample_count)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# 7. Convert labels into numbers
class_to_id = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

y = balanced_df["cleanLabel"].map(
    class_to_id
).to_numpy(dtype=np.int64)


# 8. Resize wafer maps while preserving aspect ratio
def prepare_wafer_map(wafer):

    wafer = np.asarray(wafer, dtype=np.uint8)

    height, width = wafer.shape

    side = max(height, width)

    top = (side - height) // 2
    bottom = side - height - top

    left = (side - width) // 2
    right = side - width - left

    padded = cv2.copyMakeBorder(
        wafer,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    resized = cv2.resize(
        padded,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    return resized


# 9. Convert all wafer maps
print("\nConverting wafer images...")

images = []

for index, wafer in enumerate(balanced_df["waferMap"]):

    images.append(prepare_wafer_map(wafer))

    if (index + 1) % 1000 == 0:
        print("Processed:", index + 1)


X = np.stack(images).astype(np.uint8)

# CNN expects: samples, height, width, channels
X = X[..., np.newaxis]


# 10. Split by manufacturing lot
groups = (
    balanced_df["lotName"]
    .fillna("unknown")
    .astype(str)
    .to_numpy(dtype=str)
)

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# 11. Save the processed dataset
os.makedirs("data/processed", exist_ok=True)


np.savez_compressed(
    OUTPUT_PATH,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    groups_train=groups[train_idx],
    groups_test=groups[test_idx],
    class_names=np.array(CLASS_NAMES)
)



# 12. Final report
print("\n========== PREPROCESSING REPORT ==========")

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("\nTraining distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        class_name,
        ":",
        np.sum(y_train == class_id)
    )

print("\nSaved:", OUTPUT_PATH)

print("\nPreprocessing completed!")
