
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold


# ==========================================
# 1. LOAD TRAINED CNN V2
# ==========================================

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATASET_PATH = "data/processed/wafer_dataset_64.npz"

model = tf.keras.models.load_model(MODEL_PATH)

# Build the model by passing a dummy input
dummy_input = tf.zeros((1, 64, 64, 3))

_ = model(dummy_input, training=False)

print("CNN Version 2 loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ==========================================
# 2. LOAD DATASET
# ==========================================

data = np.load(
    DATASET_PATH,
    allow_pickle=False
)

X_raw = data["X_train"]
y = data["y_train"]
groups = data["groups_train"]

class_names = data["class_names"].tolist()

print("Dataset loaded:", X_raw.shape)


# ==========================================
# 3. RECREATE VALIDATION SPLIT
# ==========================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X_raw, y, groups)
)

X_val_raw = X_raw[val_idx]
y_val = y[val_idx]


# ==========================================
# 4. FIND LAST CONVOLUTIONAL LAYER
# ==========================================

last_conv_layer = next(
    layer
    for layer in reversed(model.layers)
    if isinstance(layer, tf.keras.layers.Conv2D)
)

print("Last convolutional layer:", last_conv_layer.name)



# ==========================================
# 5. CREATE KERAS 3 COMPATIBLE GRAD-CAM MODEL
# ==========================================

# Create a fresh symbolic input
grad_input = tf.keras.Input(
    shape=(64, 64, 3)
)

x = grad_input
conv_outputs = None

# Reuse the trained CNN layers and weights
for layer in model.layers:

    x = layer(x, training=False)

    # Capture the final convolutional feature maps
    if layer.name == last_conv_layer.name:
        conv_outputs = x

# Final classification output
predictions = x

if conv_outputs is None:
    raise ValueError("Convolutional layer output not found!")

# Functional model with two outputs
grad_model = tf.keras.Model(
    inputs=grad_input,
    outputs=[
        conv_outputs,
        predictions
    ]
)

print("Grad-CAM model created successfully!")
print("Feature map shape:", conv_outputs.shape)
print("Prediction shape:", predictions.shape)


# ==========================================
# 6. GRAD-CAM FUNCTION
# ==========================================

def generate_gradcam(image, class_index=None):

    # Convert categorical wafer map into 3 channels
    image = tf.squeeze(image, axis=-1)

    image = tf.one_hot(
        tf.cast(image, tf.int32),
        depth=3
    )

    image = tf.cast(image, tf.float32)

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image,
            training=False
        )

        if class_index is None:
            class_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[:, class_index]

    # Calculate gradients
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Average gradients spatially
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :]
        * conv_outputs,
        axis=-1
    )

    # ReLU and normalization
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap[0]

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    # Resize to original wafer dimensions
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (64, 64)
    )

    return (
        heatmap.numpy().squeeze(),
        predictions.numpy()[0],
        int(class_index.numpy())
    )


# ==========================================
# 7. SELECT A WAFER FOR EXPLANATION
# ==========================================

TARGET_CLASS = "Edge-Ring"

target_id = class_names.index(TARGET_CLASS)

# Find validation samples of the selected category
candidate_indices = np.where(
    y_val == target_id
)[0]

if len(candidate_indices) == 0:
    raise ValueError(
        "No validation samples found for " + TARGET_CLASS
    )

# Prefer a correctly classified example
correct_index = None

for idx in candidate_indices:

    _, probabilities, predicted_id = generate_gradcam(
        X_val_raw[idx]
    )

    if predicted_id == target_id:
        correct_index = idx
        break

if correct_index is None:
    correct_index = candidate_indices[0]

selected_wafer = X_val_raw[correct_index]

heatmap, probabilities, predicted_id = generate_gradcam(
    selected_wafer
)

actual_id = int(y_val[correct_index])

confidence = float(probabilities[predicted_id])


# ==========================================
# 8. PRINT PREDICTION
# ==========================================

print("\n========== GRAD-CAM PREDICTION ==========")

print("Actual defect:", class_names[actual_id])

print("Predicted defect:", class_names[predicted_id])

print("Confidence:", round(confidence * 100, 2), "%")


# ==========================================
# 9. VISUALISE GRAD-CAM
# ==========================================

original = selected_wafer.squeeze()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

# Original wafer
axes[0].imshow(
    original,
    cmap="viridis",
    vmin=0,
    vmax=2,
    interpolation="nearest"
)

axes[0].set_title("Original Wafer")

# Heatmap
axes[1].imshow(
    heatmap,
    cmap="jet",
    vmin=0,
    vmax=1
)

axes[1].set_title("Grad-CAM Heatmap")

# Overlay
axes[2].imshow(
    original,
    cmap="viridis",
    vmin=0,
    vmax=2,
    interpolation="nearest"
)

axes[2].imshow(
    heatmap,
    cmap="jet",
    alpha=0.4,
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Prediction: " + class_names[predicted_id]
)

for ax in axes:
    ax.axis("off")

plt.suptitle(
    f"Grad-CAM | Confidence: {confidence * 100:.2f}%",
    fontsize=15
)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/gradcam_edge_ring.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()

print("\nGrad-CAM image saved!")
