
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# ==========================================
# 2. LOAD PROCESSED DATASET
# ==========================================

data = np.load(
    "data/processed/wafer_dataset_64.npz",
    allow_pickle=False
)

X_train_raw = data["X_train"]
X_test_raw = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

groups_train = data["groups_train"]

class_names = data["class_names"].tolist()
NUM_CLASSES = len(class_names)

print("\nOriginal training shape:", X_train_raw.shape)

# ==========================================
# 3. ONE-HOT ENCODING
# ==========================================

X_train = tf.one_hot(
    X_train_raw.squeeze(axis=-1),
    depth=3
).numpy()

X_test = tf.one_hot(
    X_test_raw.squeeze(axis=-1),
    depth=3
).numpy()

print("Encoded training shape:", X_train.shape)
print("Encoded testing shape:", X_test.shape)

# ==========================================
# 4. GROUP-AWARE TRAIN / VALIDATION SPLIT
# ==========================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X_train, y_train, groups_train)
)

X_fit = X_train[train_idx]
X_val = X_train[val_idx]

y_fit = y_train[train_idx]
y_val = y_train[val_idx]

fit_groups = set(groups_train[train_idx])
val_groups = set(groups_train[val_idx])

assert fit_groups.isdisjoint(val_groups)

print("\nTraining:", X_fit.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nValidation class counts:")

for i, name in enumerate(class_names):
    print(name, np.sum(y_val == i))

# ==========================================
# 5. CLASS WEIGHTS
# ==========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_fit),
    y=y_fit
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        np.unique(y_fit),
        weights
    )
}

print("\nClass weights:", class_weights)

# ==========================================
# 6. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(64, 64, 3)
    ),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\n========== CNN ARCHITECTURE ==========")

model.summary()

# ==========================================
# 8. CALLBACKS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/best_wafer_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# ==========================================
# 9. TRAIN CNN
# ==========================================

print("\n========== TRAINING STARTED ==========")

history = model.fit(

    X_fit,
    y_fit,

    validation_data=(X_val, y_val),

    epochs=30,
    batch_size=32,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)

# ==========================================
# 10. SAVE MODEL
# ==========================================

model.save("models/wafer_cnn_final.keras")

# ==========================================
# 11. TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/cnn_training_history.png",
    dpi=150
)

plt.show()

# ==========================================
# 12. FINAL TEST EVALUATION
# ==========================================

print("\n========== TEST EVALUATION ==========")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predictions
probabilities = model.predict(X_test)

predictions = np.argmax(
    probabilities,
    axis=1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(10, 8))

plt.imshow(cm, cmap="Blues")

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Wafer Defect Confusion Matrix")

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150
)

plt.show()

print("\nCNN training and evaluation completed!")
