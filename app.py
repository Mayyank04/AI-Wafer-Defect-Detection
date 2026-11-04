
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Wafer Defect Detection",
    page_icon="🔬",
    layout="wide"
)

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

MODEL_PATH = "models/best_wafer_cnn_v2.keras"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1020;
    color: #ffffff;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.metric-card {
    background: #151e32;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #29364f;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Build model
    model(
        tf.zeros((1, 64, 64, 3)),
        training=False
    )

    return model


model = load_model()


# ==========================================
# PREPROCESS WAFER IMAGE
# ==========================================

def preprocess_image(image):

    image = image.convert("L")

    image = image.resize(
        (64, 64),
        Image.Resampling.NEAREST
    )

    image = np.array(image)

    # Accept categorical wafer values 0, 1, 2
    if not np.all(np.isin(image, [0, 1, 2])):
        raise ValueError(
            "Image must contain categorical pixel values 0, 1, 2. "
            "Please upload a raw categorical wafer map, not a screenshot."
        )

    image = tf.one_hot(
        image.astype(np.int32),
        depth=3
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image.numpy(), np.array(
        Image.fromarray(
            np.array(image.numpy().argmax(axis=-1), dtype=np.uint8)
        )
    )


# ==========================================
# CREATE GRAD-CAM MODEL
# ==========================================

@st.cache_resource
def create_grad_model():

    last_conv = next(
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    )

    inputs = tf.keras.Input(
        shape=(64, 64, 3)
    )

    x = inputs
    conv_outputs = None

    for layer in model.layers:

        x = layer(x, training=False)

        if layer.name == last_conv.name:
            conv_outputs = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_outputs, x]
    )

    return grad_model


grad_model = create_grad_model()


# ==========================================
# GENERATE GRAD-CAM
# ==========================================

def generate_gradcam(image, class_index):

    image = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        features, predictions = grad_model(
            image,
            training=False
        )

        score = predictions[:, class_index]

    gradients = tape.gradient(
        score,
        features
    )

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

    return heatmap.numpy().squeeze()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🔬 AI Wafer Defect Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-powered semiconductor wafer inspection '
    'with Explainable AI and Grad-CAM'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Inspection Settings")

st.sidebar.info(
    "Model: CNN Version 2\n\n"
    "Architecture: CNN + BatchNorm + GAP\n\n"
    "Explainability: Grad-CAM"
)

st.sidebar.caption(
    "Input requirement: 64x64 categorical wafer maps."
)


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a wafer map",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    try:

        uploaded_image = Image.open(
            uploaded_file
        )

        wafer, original_map = preprocess_image(
            uploaded_image
        )

        # ==================================
        # PREDICTION
        # ==================================

        predictions = model.predict(
            wafer[None, ...],
            verbose=0
        )[0]

        predicted_id = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_id
        ]

        confidence = float(
            predictions[predicted_id]
        )

        # ==================================
        # GRAD-CAM
        # ==================================

        heatmap = generate_gradcam(
            wafer,
            predicted_id
        )

        # ==================================
        # METRICS
        # ==================================

        st.subheader("Inspection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Defect",
            predicted_class
        )

        col2.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        col3.metric(
            "Defect Categories",
            "9"
        )

        st.divider()

        # ==================================
        # VISUALISATIONS
        # ==================================

        st.subheader("Visual Inspection")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown("**Original Wafer**")

            st.image(
                original_map,
                clamp=True,
                use_container_width=True
            )

        with c2:

            st.markdown("**Grad-CAM Heatmap**")

            st.image(
                heatmap,
                clamp=True,
                use_container_width=True
            )

        with c3:

            st.markdown("**Prediction Overlay**")

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )

            ax.imshow(
                original_map,
                cmap="viridis",
                vmin=0,
                vmax=2,
                interpolation="nearest"
            )

            ax.imshow(
                heatmap,
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1
            )

            ax.axis("off")

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        # ==================================
        # CLASS PROBABILITIES
        # ==================================

        st.divider()

        st.subheader("Class Probability Distribution")

        sorted_indices = np.argsort(
            predictions
        )[::-1]

        for idx in sorted_indices:

            st.write(
                f"{CLASS_NAMES[idx]}: "
                f"{predictions[idx] * 100:.2f}%"
            )

            st.progress(
                float(predictions[idx])
            )

        st.success(
            f"Inspection completed: {predicted_class}"
        )

        st.caption(
            "Grad-CAM highlights influential regions, "
            "not exact defect boundaries. Model confidence "
            "is not a guarantee of prediction correctness."
        )

    except Exception as error:

        st.error(
            f"Inspection failed: {error}"
        )

else:

    st.info(
        "Upload a categorical wafer map to begin inspection."
    )
