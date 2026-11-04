
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WAFER AI | Batch Inspection",
    page_icon="🔬",
    layout="wide"
)

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATA_PATH = "data/processed/wafer_dataset_64.npz"

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
# 2. DASHBOARD DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #080D19;
    color: #E2E8F0;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #101A2B;
}

.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(
        120deg,
        #10243B,
        #103650,
        #101A2B
    );
    border: 1px solid #25445F;
}

.hero h1 {
    color: #38BDF8;
    font-size: 36px;
}

.hero p {
    color: #A8BCD1;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD CNN MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    model(
        tf.zeros((1, 64, 64, 3)),
        training=False
    )

    return model


model = load_model()


# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

@st.cache_data
def load_validation_data():

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X_train"]
    y = data["y_train"].astype(np.int32)
    groups = data["groups_train"]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    _, validation_indices = next(
        splitter.split(X, y, groups)
    )

    X_val = X[validation_indices]
    y_val = y[validation_indices]

    return X_val, y_val


X_val, y_val = load_validation_data()


# ==========================================
# 5. HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🔬 WAFER AI</h1>
    <p>Batch Wafer Inspection & Quality Analytics</p>
    <p>
        Automated CNN Classification |
        Defect Analytics |
        Semiconductor Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Batch Controls")

st.sidebar.success("CNN V2 Loaded")

st.sidebar.write(
    f"Available wafers: {len(X_val)}"
)

st.sidebar.write(
    "Input: 64 × 64 categorical wafer maps"
)

st.sidebar.write(
    "Output: 9 defect categories"
)

st.sidebar.divider()

batch_size = st.sidebar.slider(
    "Number of wafers to inspect",
    min_value=10,
    max_value=len(X_val),
    value=min(100, len(X_val)),
    step=10
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1
)

st.sidebar.caption(
    "Only validation samples are used in this "
    "version of batch inspection."
)


# ==========================================
# 7. BATCH INSPECTION FUNCTION
# ==========================================

def inspect_batch(batch_X, batch_y):

    # Remove the single-channel dimension
    if batch_X.ndim == 4:
        batch_X = batch_X.squeeze(-1)

    # Convert categorical pixels into 3 channels
    encoded = tf.one_hot(
        tf.cast(batch_X, tf.int32),
        depth=3
    )

    encoded = tf.cast(
        encoded,
        tf.float32
    )

    # Continue with model.predict(...)

    # Predict all wafers in batches
    probabilities = model.predict(
        encoded,
        batch_size=32,
        verbose=0
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    results = pd.DataFrame({

        "Wafer_ID": [
            f"WAFER_{i+1:04d}"
            for i in range(len(batch_y))
        ],

        "Actual_Defect": [
            CLASS_NAMES[i]
            for i in batch_y
        ],

        "Predicted_Defect": [
            CLASS_NAMES[i]
            for i in predicted_labels
        ],

        "Confidence": confidence,

        "Correct": (
            batch_y == predicted_labels
        )

    })

    # Add individual class probabilities
    for i, class_name in enumerate(CLASS_NAMES):

        results[
            f"Probability_{class_name}"
        ] = probabilities[:, i]

    return results, probabilities, predicted_labels


# ==========================================
# 8. RUN BATCH INSPECTION
# ==========================================

st.subheader("Batch Inspection Workspace")

st.write(
    "Select a batch size and run the CNN "
    "on randomly selected validation wafers."
)

if st.button(
    "🚀 Start Batch Inspection",
    type="primary",
    use_container_width=True
):

    rng = np.random.default_rng(
        int(random_seed)
    )

    selected_indices = rng.choice(
        len(X_val),
        size=batch_size,
        replace=False
    )

    batch_X = X_val[selected_indices]
    batch_y = y_val[selected_indices]

    with st.spinner(
        f"Inspecting {batch_size} wafers..."
    ):

        results, probabilities, predicted_labels = (
            inspect_batch(batch_X, batch_y)
        )

        accuracy = accuracy_score(
            batch_y,
            predicted_labels
        )

        macro_f1 = f1_score(
            batch_y,
            predicted_labels,
            labels=np.arange(len(CLASS_NAMES)),
            average="macro",
            zero_division=0
        )

        correct_count = int(
            results["Correct"].sum()
        )

        defective_count = int(
            (results["Predicted_Defect"] != "none").sum()
        )

    # ======================================
    # 9. KEY METRICS
    # ======================================

    st.divider()

    st.subheader("Inspection Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Wafers Inspected",
        len(results)
    )

    c2.metric(
        "Correct Predictions",
        correct_count
    )

    c3.metric(
        "Defective Predictions",
        defective_count
    )

    c4.metric(
        "Batch Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.metric(
        "Macro F1 Score",
        f"{macro_f1:.4f}"
    )

    st.caption(
        "Defective predictions exclude the 'none' class. "
        "These are model predictions, not verified physical "
        "defect counts."
    )

    # ======================================
    # 10. DEFECT DISTRIBUTION
    # ======================================

    st.divider()

    st.subheader("Defect Distribution")

    actual_counts = (
        results["Actual_Defect"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    predicted_counts = (
        results["Predicted_Defect"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    distribution = pd.DataFrame({

        "Actual": actual_counts,

        "Predicted": predicted_counts

    })

    st.bar_chart(
        distribution
    )

    # ======================================
    # 11. CONFUSION MATRIX
    # ======================================

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        batch_y,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES))
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_xticks(
        range(len(CLASS_NAMES))
    )

    ax.set_yticks(
        range(len(CLASS_NAMES))
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title("Wafer Defect Confusion Matrix")

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ======================================
    # 12. CLASSIFICATION REPORT
    # ======================================

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        batch_y,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # ======================================
    # 13. BATCH RESULTS TABLE
    # ======================================

    st.divider()

    st.subheader("Individual Wafer Results")

    st.dataframe(
        results,
        use_container_width=True,
        height=400
    )

    # ======================================
    # 14. DOWNLOAD CSV
    # ======================================

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Batch Inspection CSV",
        data=csv,
        file_name="wafer_batch_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(
        f"Batch inspection completed! "
        f"{len(results)} wafers analysed."
    )

    st.caption(
        "Wafer IDs are generated for this batch and "
        "are not original manufacturing identifiers."
    )

else:

    st.info(
        "Configure your batch size in the sidebar "
        "and click Start Batch Inspection."
    )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>WAFER AI | Batch Inspection V1 | "
    "CNN + Computer Vision</center>",
    unsafe_allow_html=True
)
