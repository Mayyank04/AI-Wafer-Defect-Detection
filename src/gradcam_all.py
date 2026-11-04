
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold

# 1. Load trained model
model = tf.keras.models.load_model(
    "models/best_wafer_cnn_v2.keras"
)

_ = model(tf.zeros((1, 64, 64, 3)))

# 2. Load dataset
data = np.load(
    "data/processed/wafer_dataset_64.npz",
    allow_pickle=False
)

X_raw = data["X_train"]
y = data["y_train"]
groups = data["groups_train"]

class_names = data["class_names"].tolist()

# 3. Recreate validation split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

_, val_idx = next(
    splitter.split(X_raw, y, groups)
)

X_val = X_raw[val_idx]
y_val = y[val_idx]

# 4. Identify final convolution layer
last_conv = next(
    layer for layer in reversed(model.layers)
    if isinstance(layer, tf.keras.layers.Conv2D)
)

# 5. Build Grad-CAM model
grad_input = tf.keras.Input(shape=(64, 64, 3))

x = grad_input
conv_outputs = None

for layer in model.layers:

    x = layer(x, training=False)

    if layer.name == last_conv.name:
        conv_outputs = x

grad_model = tf.keras.Model(
    inputs=grad_input,
    outputs=[conv_outputs, x]
)


# 6. Grad-CAM function
def generate_gradcam(wafer):

    image = tf.one_hot(
        tf.cast(wafer.squeeze(), tf.int32),
        depth=3
    )

    image = tf.cast(image, tf.float32)
    image = image[tf.newaxis, ...]

    with tf.GradientTape() as tape:

        features, predictions = grad_model(
            image,
            training=False
        )

        class_id = tf.argmax(
            predictions[0]
        )

        score = predictions[:, class_id]

    gradients = tape.gradient(score, features)

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        weights[:, None, None, :] * features,
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)[0]

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., None],
        (64, 64)
    )

    return (
        heatmap.numpy().squeeze(),
        predictions.numpy()[0],
        int(class_id.numpy())
    )


# 7. Generate visualisations for all classes
os.makedirs("outputs/gradcam_all", exist_ok=True)

print("\n========== ALL CLASS GRAD-CAM ==========")

for class_id, class_name in enumerate(class_names):

    candidates = np.where(y_val == class_id)[0]

    if len(candidates) == 0:
        print("No validation sample:", class_name)
        continue

    # Prefer correctly classified example
    selected_idx = candidates[0]

    for idx in candidates:

        _, probabilities, predicted = generate_gradcam(
            X_val[idx]
        )

        if predicted == class_id:
            selected_idx = idx
            break

    wafer = X_val[selected_idx]

    heatmap, probabilities, predicted = generate_gradcam(
        wafer
    )

    confidence = float(probabilities[predicted])

    original = wafer.squeeze()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(
        original,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    axes[0].set_title("Original Wafer")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM")

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
        "Prediction: " + class_names[predicted]
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Actual: {class_name} | "
        f"Prediction: {class_names[predicted]} | "
        f"Confidence: {confidence:.2%}"
    )

    plt.tight_layout()

    filename = (
        "outputs/gradcam_all/"
        + class_name.replace("-", "_")
        + ".png"
    )

    plt.savefig(filename, dpi=150)
    plt.close()

    print(
        class_name,
        "->",
        class_names[predicted],
        "| Confidence:",
        round(confidence * 100, 2),
        "%"
    )

print("\nAll Grad-CAM visualisations generated!")
