
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())


# ==========================================
# 2. LOAD DATASET
# ==========================================

data = np.load(
    "data/processed/wafer_dataset_64.npz",
    allow_pickle=False
)

X_raw = data["X_train"]
X_test_raw = data["X_test"]

y = data["y_train"].astype(np.int32)
y_test = data["y_test"].astype(np.int32)

groups = data["groups_train"]

class_names = data["class_names"].tolist()
NUM_CLASSES = len(class_names)

print("\nRaw training shape:", X_raw.shape)


# ==========================================
# 3. GROUP-AWARE VALIDATION SPLIT
# ==========================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_idx, val_idx = next(
    splitter.split(X_raw, y, groups)
)

X_fit_raw = X_raw[train_idx]
X_val_raw = X_raw[val_idx]

y_fit = y[train_idx]
y_val = y[val_idx]

fit_groups = set(groups[train_idx].tolist())
val_groups = set(groups[val_idx].tolist())

assert fit_groups.isdisjoint(val_groups)

print("Training samples:", len(y_fit))
print("Validation samples:", len(y_val))
print("Overlapping validation lots:", len(
    fit_groups.intersection(val_groups)
))


# ==========================================
# 4. CLASS WEIGHTS
# ==========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_fit
)

weight_vector = tf.constant(
    weights,
    dtype=tf.float32
)

print("\nClass weights:")

for name, weight in zip(class_names, weights):
    print(name, round(float(weight), 3))


# ==========================================
# 5. DATA PREPROCESSING
# ==========================================

def encode_image(image):

    image = tf.squeeze(image, axis=-1)

    image = tf.one_hot(
        tf.cast(image, tf.int32),
        depth=3
    )

    return tf.cast(image, tf.float32)


# ==========================================
# 6. DATA AUGMENTATION
# ==========================================

def augment_image(image, label):

    image = encode_image(image)

    # Random rotation in multiples of 90 degrees
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32
    )

    image = tf.image.rot90(image, k)

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random vertical flip
    image = tf.image.random_flip_up_down(image)

    sample_weight = tf.gather(
        weight_vector,
        label
    )

    return image, label, sample_weight


def prepare_validation(image, label):

    image = encode_image(image)

    return image, label


# ==========================================
# 7. CREATE TF.DATA PIPELINES
# ==========================================

BATCH_SIZE = 32

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_fit_raw, y_fit)
)

train_ds = train_ds.shuffle(
    buffer_size=len(y_fit),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val_raw, y_val)
)

val_ds = val_ds.map(
    prepare_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ==========================================
# 8. BUILD CNN VERSION 2
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(64, 64, 3)
    ),

    # BLOCK 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # BLOCK 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # BLOCK 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # GLOBAL AVERAGE POOLING
    tf.keras.layers.GlobalAveragePooling2D(),

    # CLASSIFICATION
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
# 9. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\n========== CNN V2 ARCHITECTURE ==========")

model.summary()


# ==========================================
# 10. CALLBACKS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/best_wafer_cnn_v2.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\n========== CNN V2 TRAINING ==========")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=30,

    callbacks=callbacks,

    verbose=1
)


# ==========================================
# 12. SAVE MODEL
# ==========================================

model.save(
    "models/wafer_cnn_v2_final.keras"
)

print("\nCNN V2 saved successfully!")


# ==========================================
# 13. TRAINING HISTORY
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.title("CNN V2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.title("CNN V2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/cnn_v2_training_history.png",
    dpi=150
)

plt.show()


# ==========================================
# 14. FINAL TEST EVALUATION
# ==========================================

print("\n========== CNN V2 TEST EVALUATION ==========")

# Encode held-out test images without augmentation
X_test = tf.one_hot(
    tf.squeeze(X_test_raw, axis=-1),
    depth=3
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

probabilities = model.predict(
    X_test,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print("\nTest Accuracy:", round(test_accuracy, 4))
print("Test Macro F1:", round(macro_f1, 4))

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


# ==========================================
# 15. CONFUSION MATRIX
# ==========================================

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

plt.title("CNN V2 Confusion Matrix")

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "outputs/cnn_v2_confusion_matrix.png",
    dpi=150
)

plt.show()

print("\nCNN V2 completed successfully!")
