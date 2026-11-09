
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATASET_PATH = "data/processed/wafer_dataset_64.npz"

OUTPUT_DIR = "outputs/gradcam_errors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 32

print("\n========== FULL VALIDATION ERROR INVESTIGATOR ==========\n")


# =====================================================
# 1. LOAD MODEL AND DATASET
# =====================================================

print("[1/6] Loading CNN V2 and dataset...")

model = tf.keras.models.load_model(MODEL_PATH)

dataset = np.load(DATASET_PATH, allow_pickle=False)

X_train = dataset["X_train"]
y_train = dataset["y_train"]
groups_train = dataset["groups_train"]

class_names = [
    x.decode() if isinstance(x, bytes) else str(x)
    for x in dataset["class_names"].tolist()
]

if y_train.ndim > 1:
    y_train = np.argmax(y_train, axis=1)

y_train = y_train.astype(int)

print("Model loaded:", MODEL_PATH)
print("Training pool:", X_train.shape)
print("Classes:", class_names)


# =====================================================
# 2. RECREATE ORIGINAL VALIDATION SPLIT
# =====================================================

print("\n[2/6] Recreating group-aware validation split...")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_indices, validation_indices = next(
    splitter.split(X_train, y_train, groups_train)
)

X_val = X_train[validation_indices]
y_val = y_train[validation_indices]

print("Training subset:", len(train_indices))
print("Validation wafers:", len(X_val))

# Check that no lot overlaps between training and validation
train_groups = set(groups_train[train_indices])
val_groups = set(groups_train[validation_indices])

assert train_groups.isdisjoint(val_groups), (
    "Data leakage detected: overlapping lots!"
)

print("Group leakage check: PASSED")


# =====================================================
# 3. PREDICT ALL VALIDATION WAFERS
# =====================================================

print("\n[3/6] Predicting complete validation dataset...")

X_val_onehot = tf.one_hot(
    tf.cast(np.squeeze(X_val, axis=-1), tf.int32),
    depth=3
)

predictions = model.predict(
    X_val_onehot,
    batch_size=BATCH_SIZE,
    verbose=1
)

predicted_labels = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)

errors_mask = predicted_labels != y_val

error_positions = np.where(errors_mask)[0]

print("\nTotal validation wafers:", len(y_val))
print("Correct predictions:", int(np.sum(~errors_mask)))
print("Incorrect predictions:", len(error_positions))
print("Validation accuracy:", f"{np.mean(~errors_mask):.2%}")


# =====================================================
# 4. EXPORT COMPLETE PREDICTION RESULTS
# =====================================================

print("\n[4/6] Generating performance reports...")

all_predictions_path = os.path.join(
    OUTPUT_DIR,
    "validation_predictions.csv"
)

with open(all_predictions_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Dataset_Index",
        "Actual_Defect",
        "Predicted_Defect",
        "Confidence",
        "Actual_Class_Probability",
        "Prediction_Error"
    ])

    for i in range(len(y_val)):

        actual_id = int(y_val[i])
        predicted_id = int(predicted_labels[i])

        writer.writerow([
            int(validation_indices[i]),
            class_names[actual_id],
            class_names[predicted_id],
            float(confidence[i]),
            float(predictions[i, actual_id]),
            bool(errors_mask[i])
        ])


# Classification report

report = classification_report(
    y_val,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

import pandas as pd

pd.DataFrame(report).transpose().to_csv(report_path)


# Confusion matrix

cm = confusion_matrix(
    y_val,
    predicted_labels,
    labels=np.arange(len(class_names))
)

fig, ax = plt.subplots(figsize=(11, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=True
)

ax.set_title("CNN V2 - Full Validation Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("Prediction CSV saved.")
print("Classification report saved.")
print("Confusion matrix saved.")


# =====================================================
# 5. BUILD GRAD-CAM MODEL
# =====================================================

print("\n[5/6] Building Grad-CAM model...")

last_conv = next(
    (
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    ),
    None
)

if last_conv is None:
    raise ValueError("No Conv2D layer found!")

grad_input = tf.keras.Input(shape=(64, 64, 3))

x = grad_input
conv_outputs = None

for layer in model.layers:

    x = layer(x, training=False)

    if layer.name == last_conv.name:
        conv_outputs = x

if conv_outputs is None:
    raise ValueError("Could not capture convolutional features.")

grad_model = tf.keras.Model(
    inputs=grad_input,
    outputs=[conv_outputs, x]
)

print("Last convolutional layer:", last_conv.name)
print("Grad-CAM model ready!")


def generate_gradcam(image, class_index):

    image = tf.convert_to_tensor(
        image[tf.newaxis, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        features, predictions_for_image = grad_model(
            image,
            training=False
        )

        score = predictions_for_image[:, class_index]

    gradients = tape.gradient(score, features)

    if gradients is None:
        raise ValueError("Grad-CAM gradients unavailable.")

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :] * features,
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)[0]

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        maximum
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (64, 64)
    )

    return heatmap.numpy().squeeze()


# =====================================================
# 6. INVESTIGATE ALL MISCLASSIFICATIONS
# =====================================================

print("\n[6/6] Generating Grad-CAM explanations...\n")

summary = []

for count, position in enumerate(error_positions, start=1):

    wafer = np.squeeze(X_val[position])

    actual_id = int(y_val[position])
    predicted_id = int(predicted_labels[position])

    actual_name = class_names[actual_id]
    predicted_name = class_names[predicted_id]

    dataset_index = int(validation_indices[position])

    prediction_confidence = float(
        predictions[position, predicted_id]
    )

    actual_probability = float(
        predictions[position, actual_id]
    )

    # Generate class-specific Grad-CAM
    predicted_heatmap = generate_gradcam(
        X_val_onehot[position].numpy(),
        predicted_id
    )

    actual_heatmap = generate_gradcam(
        X_val_onehot[position].numpy(),
        actual_id
    )

    # Create visualization
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    axes[0, 0].imshow(
        wafer,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    axes[0, 0].set_title("Original Wafer")

    axes[0, 1].imshow(
        predicted_heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title(
        f"Predicted Activation: {predicted_name}"
    )

    axes[1, 0].imshow(
        wafer,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )

    axes[1, 0].imshow(
        predicted_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1, 0].set_title("Predicted Class Grad-CAM Overlay")

    axes[1, 1].imshow(
        wafer,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )

    axes[1, 1].imshow(
        actual_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1, 1].set_title(
        f"Actual Class Activation: {actual_name}"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Wafer {dataset_index} | "
        f"Actual: {actual_name} | "
        f"Predicted: {predicted_name} | "
        f"Confidence: {prediction_confidence:.2%}",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    filename = f"wafer_{dataset_index}.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    summary.append({
        "Dataset_Index": dataset_index,
        "Actual_Defect": actual_name,
        "Predicted_Defect": predicted_name,
        "Confidence": prediction_confidence,
        "Actual_Class_Probability": actual_probability,
        "Prediction_Error": True,
        "GradCAM_Image": filename
    })

    print(
        f"Processed {count}/{len(error_positions)} | "
        f"Wafer {dataset_index} | "
        f"{actual_name} -> {predicted_name}"
    )


# Export error summary

error_csv_path = os.path.join(
    OUTPUT_DIR,
    "error_summary.csv"
)

columns = [
    "Dataset_Index",
    "Actual_Defect",
    "Predicted_Defect",
    "Confidence",
    "Actual_Class_Probability",
    "Prediction_Error",
    "GradCAM_Image"
]

pd.DataFrame(summary, columns=columns).to_csv(
    error_csv_path,
    index=False
)


# =====================================================
# FINAL RESULTS
# =====================================================

print("\n==========================================")
print("FULL VALIDATION INVESTIGATION COMPLETED")
print("==========================================")

print("Total validation wafers:", len(y_val))
print("Correct predictions:", int(np.sum(~errors_mask)))
print("Incorrect predictions:", len(error_positions))

print(
    "Validation accuracy:",
    f"{np.mean(~errors_mask):.2%}"
)

print(
    "High-confidence errors (>=90%):",
    sum(item["Confidence"] >= 0.90 for item in summary)
)

print("Grad-CAM images:", len(summary))
print("Output directory:", OUTPUT_DIR)

print("\nGenerated files:")
print("- validation_predictions.csv")
print("- error_summary.csv")
print("- classification_report.csv")
print("- confusion_matrix.png")

print("\nAll investigations completed!")
