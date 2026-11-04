
import numpy as np
import matplotlib.pyplot as plt
import os

# Load processed dataset
data = np.load(
    "data/processed/wafer_dataset_64.npz",
    allow_pickle=False
)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

class_names = data["class_names"].tolist()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print("\nUnique pixel values:", np.unique(X_train))

# Create output folder
os.makedirs("outputs", exist_ok=True)


# VISUALISATION 1: One image per defect category
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for class_id, ax in enumerate(axes.flat):

    indices = np.where(y_train == class_id)[0]

    if len(indices) == 0:
        ax.set_title(class_names[class_id] + " (missing)")
        ax.axis("off")
        continue

    sample_index = indices[0]

    wafer = X_train[sample_index].squeeze()

    ax.imshow(
        wafer,
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=2
    )

    ax.set_title(class_names[class_id])
    ax.axis("off")

plt.suptitle(
    "Processed Wafer Defect Images",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "outputs/processed_wafer_examples.png",
    dpi=150
)

plt.show()


# VISUALISATION 2: Class distribution
train_counts = [
    np.sum(y_train == i)
    for i in range(len(class_names))
]

test_counts = [
    np.sum(y_test == i)
    for i in range(len(class_names))
]

x = np.arange(len(class_names))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    train_counts,
    width,
    label="Training"
)

plt.bar(
    x + width / 2,
    test_counts,
    width,
    label="Testing"
)

plt.xticks(
    x,
    class_names,
    rotation=35,
    ha="right"
)

plt.ylabel("Number of wafer images")
plt.title("Training vs Testing Class Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/class_distribution.png",
    dpi=150
)

plt.show()

print("\nVisualisation completed!")
