
import numpy as np

data = np.load(
    "data/processed/wafer_dataset_64.npz",
    allow_pickle=False
)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

groups_train = data["groups_train"]
groups_test = data["groups_test"]

class_names = data["class_names"]

print("========== DATASET VALIDATION ==========")

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# Check image dimensions
assert X_train.shape[1:] == (64, 64, 1)
assert X_test.shape[1:] == (64, 64, 1)

# Check all classes exist in both sets
assert set(np.unique(y_train)) == set(range(len(class_names)))
assert set(np.unique(y_test)) == set(range(len(class_names)))

# Check lot separation
train_groups = set(groups_train.tolist())
test_groups = set(groups_test.tolist())

overlap = train_groups.intersection(test_groups)

print("\nTraining lots:", len(train_groups))
print("Testing lots:", len(test_groups))
print("Overlapping lots:", len(overlap))

assert len(overlap) == 0, "Manufacturing lot leakage detected!"

print("\nAll validation checks passed!")
