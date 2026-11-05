
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATASET_PATH = "data/processed/wafer_dataset_64.npz"
ERROR_PATH = "outputs/error_analysis.npz"

OUTPUT_DIR = "outputs/gradcam_errors"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n========== WAFER AI ERROR INVESTIGATOR ==========\n")


# =====================================================
# 1. LOAD CNN V2
# =====================================================

print("[1/5] Loading CNN V2...")

model = tf.keras.models.load_model(MODEL_PATH)

_ = model(tf.zeros((1, 64, 64, 3)), training=False)

print("Model loaded:", MODEL_PATH)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# =====================================================
# 2. LOAD CLASS NAMES AND ERROR DATA
# =====================================================

print("\n[2/5] Loading error analysis data...")

dataset = np.load(DATASET_PATH, allow_pickle=False)

class_names = [
    x.decode() if isinstance(x, bytes) else str(x)
    for x in dataset["class_names"].tolist()
]

errors = np.load(ERROR_PATH, allow_pickle=False)

wafer_maps = errors["wafer_maps"]
actual_labels = errors["actual_labels"]
predicted_labels = errors["predicted_labels"]
dataset_indices = errors["dataset_indices"]
probabilities = errors["probabilities"]

print("Misclassified wafers:", len(wafer_maps))
print("Classes:", class_names)

assert len(class_names) == model.output_shape[-1]
assert len(wafer_maps) == len(actual_labels)


# =====================================================
# 3. BUILD GRAD-CAM MODEL
# =====================================================

print("\n[3/5] Building Grad-CAM model...")

last_conv = next(
    (
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    ),
    None
)

if last_conv is None:
    raise ValueError("No Conv2D layer found in model!")

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


# =====================================================
# 4. GRAD-CAM GENERATION
# =====================================================

def generate_gradcam(wafer, class_index):

    wafer = np.asarray(wafer).squeeze()

    if wafer.shape != (64, 64):
        raise ValueError(
            f"Unexpected wafer shape: {wafer.shape}"
        )

    image = tf.one_hot(
        tf.cast(wafer, tf.int32),
        depth=3
    )

    image = tf.cast(image, tf.float32)
    image = image[tf.newaxis, ...]

    with tf.GradientTape() as tape:

        features, predictions = grad_model(
            image,
            training=False
        )

        score = predictions[:, class_index]

    gradients = tape.gradient(score, features)

    if gradients is None:
        raise ValueError(
            "Gradients unavailable. Check the model architecture."
        )

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
# 5. INVESTIGATE ALL MISCLASSIFIED WAFERS
# =====================================================

print("\n[4/5] Generating Grad-CAM explanations...\n")

summary = []

for i in range(len(wafer_maps)):

    wafer = wafer_maps[i].squeeze()

    actual_id = int(actual_labels[i])
    predicted_id = int(predicted_labels[i])

    actual_name = class_names[actual_id]
    predicted_name = class_names[predicted_id]

    prediction_confidence = float(
        probabilities[i, predicted_id]
    )

    # Generate predicted-class activation
    predicted_heatmap = generate_gradcam(
        wafer,
        predicted_id
    )

    # Generate actual-class activation
    actual_heatmap = generate_gradcam(
        wafer,
        actual_id
    )

    # =================================================
    # CREATE VISUALIZATION
    # =================================================

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    # Original wafer
    axes[0, 0].imshow(
        wafer,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )

    axes[0, 0].set_title(
        f"Original Wafer\nDataset Index: {dataset_indices[i]}"
    )

    # Predicted class heatmap
    axes[0, 1].imshow(
        predicted_heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[0, 1].set_title(
        f"Predicted Activation: {predicted_name}"
    )

    # Predicted class overlay
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

    axes[1, 0].set_title(
        "Predicted Class Grad-CAM Overlay"
    )

    # Actual class overlay
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
        f"Wafer {dataset_indices[i]} | "
        f"Actual: {actual_name} | "
        f"Predicted: {predicted_name} | "
        f"Confidence: {prediction_confidence:.2%}",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    filename = f"wafer_{dataset_indices[i]}.png"

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Store investigation results
    summary.append({
        "Dataset_Index": int(dataset_indices[i]),
        "Actual_Defect": actual_name,
        "Predicted_Defect": predicted_name,
        "Confidence": prediction_confidence,
        "Actual_Class_Probability": float(
            probabilities[i, actual_id]
        ),
        "Prediction_Error": actual_name != predicted_name,
        "GradCAM_Image": filename
    })

    print(
        f"Processed {i + 1}/{len(wafer_maps)} | "
        f"Wafer {dataset_indices[i]} | "
        f"{actual_name} -> {predicted_name}"
    )


# =====================================================
# 6. EXPORT ERROR INVESTIGATION REPORT
# =====================================================

print("\n[5/5] Exporting investigation report...")

csv_path = os.path.join(
    OUTPUT_DIR,
    "error_summary.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


# =====================================================
# 7. FINAL SUMMARY
# =====================================================

print("\n==========================================")
print("GRAD-CAM INVESTIGATION COMPLETED")
print("==========================================")

print("Total investigated:", len(summary))

print("Images saved:", OUTPUT_DIR)

print("CSV report:", csv_path)

high_confidence_errors = sum(
    item["Confidence"] >= 0.90
    for item in summary
)

print(
    "High-confidence errors (>=90%):",
    high_confidence_errors
)

print("\nAll investigations completed successfully!")
