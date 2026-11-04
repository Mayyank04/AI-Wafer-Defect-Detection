
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WAFER AI | Inspection",
    page_icon="🔬",
    layout="wide"
)

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATA_PATH = "data/processed/wafer_dataset_64.npz"

SEED = 42

# ==========================================
# 2. CLASS NAMES
# ==========================================

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


# ==========================================
# 3. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #080D19;
    color: #E2E8F0;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #101A2B;
}

.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(
        120deg,
        #10243B,
        #102E46,
        #101A2B
    );
    border: 1px solid #25445F;
    margin-bottom: 25px;
}

.hero h1 {
    color: #38BDF8;
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    color: #A8BCD1;
}

.metric-card {
    background: #111E31;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263B55;
}

.section-title {
    color: #38BDF8;
    font-size: 21px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Build model
    loaded_model(
        tf.zeros((1, 64, 64, 3)),
        training=False
    )

    return loaded_model


model = load_model()


# ==========================================
# 5. LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():

    # This is our own trusted local dataset.
    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X_train"]
    y = data["y_train"].astype(np.int32)
    groups = data["groups_train"]

    # Reproduce the original group-aware split
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    _, val_idx = next(
        splitter.split(X, y, groups)
    )

    return X[val_idx], y[val_idx]


X_val, y_val = load_dataset()


# ==========================================
# 6. CREATE GRAD-CAM MODEL
# ==========================================

@st.cache_resource
def create_gradcam_model():

    last_conv = next(
        layer for layer in reversed(model.layers)
        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        )
    )

    inputs = tf.keras.Input(
        shape=(64, 64, 3)
    )

    x = inputs
    feature_maps = None

    for layer in model.layers:

        x = layer(x, training=False)

        if layer.name == last_conv.name:
            feature_maps = x

    return tf.keras.Model(
        inputs=inputs,
        outputs=[feature_maps, x]
    )


grad_model = create_gradcam_model()


# ==========================================
# 7. IMAGE PREPROCESSING
# ==========================================

def encode_wafer(wafer):

    wafer = np.asarray(wafer)

    if wafer.shape == (64, 64, 1):
        wafer = wafer.squeeze(-1)

    if wafer.shape != (64, 64):
        raise ValueError(
            "Wafer must have shape 64 x 64."
        )

    if not np.all(
        np.isin(wafer, [0, 1, 2])
    ):
        raise ValueError(
            "Invalid wafer values. "
            "Expected 0, 1 and 2."
        )

    encoded = tf.one_hot(
        wafer.astype(np.int32),
        depth=3
    )

    return tf.cast(
        encoded,
        tf.float32
    ).numpy()


# ==========================================
# 8. PREDICTION
# ==========================================

def predict_wafer(wafer):

    probabilities = model.predict(
        wafer[None, ...],
        verbose=0
    )[0]

    predicted_id = int(
        np.argmax(probabilities)
    )

    return predicted_id, probabilities


# ==========================================
# 9. GRAD-CAM
# ==========================================

def generate_gradcam(wafer, class_id):

    image = tf.convert_to_tensor(
        wafer[None, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        features, predictions = grad_model(
            image,
            training=False
        )

        score = predictions[:, class_id]

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
# 10. HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🔬 WAFER AI</h1>
    <p>Intelligent Semiconductor Defect Inspection</p>
    <p>CNN Classification • Explainable AI • Grad-CAM</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 11. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Inspection Controls")

st.sidebar.success("CNN V2 Model Loaded")

st.sidebar.markdown("""
**Model Information**

- Architecture: CNN
- Batch Normalization: Enabled
- Global Average Pooling: Enabled
- Input: 64 × 64
- Output: 9 classes
- Explainability: Grad-CAM
""")

st.sidebar.divider()

source = st.sidebar.radio(
    "Select input source",
    [
        "Dataset Sample",
        "Upload Wafer Image"
    ]
)


# ==========================================
# 12. SELECT WAFER
# ==========================================

wafer = None
actual_label = None

if source == "Dataset Sample":

    selected_class = st.selectbox(
        "Select wafer defect category",
        CLASS_NAMES
    )

    class_id = CLASS_NAMES.index(
        selected_class
    )

    candidates = np.where(
        y_val == class_id
    )[0]

    if len(candidates) == 0:

        st.warning(
            "No validation samples found."
        )

        st.stop()

    selected_sample = st.selectbox(
        "Select wafer sample",
        range(len(candidates)),
        format_func=lambda i: (
            f"Validation wafer {i + 1}"
        )
    )

    sample_idx = candidates[selected_sample]

    raw_wafer = X_val[sample_idx]

    wafer = encode_wafer(raw_wafer)

    actual_label = CLASS_NAMES[
        int(y_val[sample_idx])
    ]

else:

    uploaded_file = st.file_uploader(
        "Upload categorical wafer image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("L")

            image = image.resize(
                (64, 64),
                Image.Resampling.NEAREST
            )

            raw = np.asarray(image)

            if not np.all(
                np.isin(raw, [0, 1, 2])
            ):
                st.error(
                    "This image is not a categorical "
                    "wafer map. Expected pixel values "
                    "0, 1 and 2."
                )

                st.stop()

            wafer = encode_wafer(raw)

        except Exception as error:

            st.error(str(error))
            st.stop()


# ==========================================
# 13. RUN INSPECTION
# ==========================================

if wafer is not None:

    st.divider()

    with st.spinner(
        "AI engine analysing wafer..."
    ):

        predicted_id, probabilities = predict_wafer(
            wafer
        )

        predicted_class = CLASS_NAMES[
            predicted_id
        ]

        confidence = float(
            probabilities[predicted_id]
        )

        heatmap = generate_gradcam(
            wafer,
            predicted_id
        )

    # ======================================
    # 14. RESULTS
    # ======================================

    st.markdown(
        '<p class="section-title">Inspection Results</p>',
        unsafe_allow_html=True
    )

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

    if actual_label is not None:

        if actual_label == predicted_class:

            st.success(
                f"Correct classification: {actual_label}"
            )

        else:

            st.warning(
                f"Actual: {actual_label} | "
                f"Predicted: {predicted_class}"
            )

    st.divider()

    # ======================================
    # 15. VISUAL INSPECTION
    # ======================================

    st.markdown(
        '<p class="section-title">Visual Inspection</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    original = wafer.argmax(axis=-1)

    with col1:

        st.markdown("**Original Wafer**")

        st.image(
            original,
            clamp=True,
            use_container_width=True
        )

    with col2:

        st.markdown("**Grad-CAM Heatmap**")

        st.image(
            heatmap,
            clamp=True,
            use_container_width=True
        )

    with col3:

        st.markdown("**Prediction Overlay**")

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ax.imshow(
            original,
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

    st.divider()

    # ======================================
    # 16. PROBABILITY ANALYSIS
    # ======================================

    st.markdown(
        '<p class="section-title">'
        'Defect Probability Analysis'
        '</p>',
        unsafe_allow_html=True
    )

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    for idx in sorted_indices:

        value = float(
            probabilities[idx]
        )

        st.write(
            f"**{CLASS_NAMES[idx]}** — "
            f"{value * 100:.2f}%"
        )

        st.progress(
            value
        )

    st.divider()

    st.caption(
        "Grad-CAM highlights regions influencing "
        "the selected prediction. It is not a "
        "pixel-level defect segmentation mask. "
        "Confidence is not a guarantee of correctness."
    )

else:

    st.info(
        "Choose a validation wafer or upload "
        "a categorical wafer map to begin."
    )


# ==========================================
# 17. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>WAFER AI | Semiconductor Intelligence "
    "Platform | CNN + Explainable AI</center>",
    unsafe_allow_html=True
)
