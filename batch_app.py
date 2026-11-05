import os

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
    classification_report,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WAFER AI | Batch Inspection",
    page_icon="🔬",
    layout="wide",
)

MODEL_PATH = "models/best_wafer_cnn_v2.keras"
DATA_PATH = "data/processed/wafer_dataset_64.npz"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "none",
    "Center",
    "Donut",
    "Edge-Ring",
    "Edge-Loc",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]


# ============================================================
# 2. DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #080D19;
        color: #E2E8F0;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
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
        margin-bottom: 8px;
    }

    .hero p {
        color: #A8BCD1;
    }

    div[data-testid="stMetric"] {
        background: #101A2B;
        border: 1px solid #23364D;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD CNN V2 MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    # Build the model explicitly before inference.
    loaded_model(
        tf.zeros((1, 64, 64, 3), dtype=tf.float32),
        training=False,
    )

    if loaded_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Model output count does not match CLASS_NAMES. "
            f"Expected {len(CLASS_NAMES)}, "
            f"got {loaded_model.output_shape[-1]}."
        )

    return loaded_model


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load CNN V2: {exc}")
    st.stop()


# ============================================================
# 4. LOAD VALIDATION DATA AND PRESERVE ORIGINAL INDICES
# ============================================================

@st.cache_data
def load_validation_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:
        required_keys = {"X_train", "y_train", "groups_train"}
        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            raise KeyError(
                f"Dataset is missing required keys: {sorted(missing_keys)}"
            )

        X = data["X_train"]
        y = data["y_train"].astype(np.int32)
        groups = data["groups_train"]

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError(
            "X_train, y_train and groups_train must have equal lengths."
        )

    if X.ndim not in (3, 4):
        raise ValueError(
            f"Unexpected image array shape: {X.shape}"
        )

    if X.ndim == 4 and X.shape[-1] != 1:
        raise ValueError(
            f"Expected one categorical channel, got shape {X.shape}"
        )

    if np.any(y < 0) or np.any(y >= len(CLASS_NAMES)):
        raise ValueError(
            "Labels contain values outside the configured class range."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    _, validation_indices = next(
        splitter.split(X, y, groups)
    )

    X_val = X[validation_indices]
    y_val = y[validation_indices]

    # Keep original positions in X_train. These are dataset row indices,
    # not verified manufacturing wafer identifiers.
    return X_val, y_val, validation_indices


try:
    X_val, y_val, val_indices = load_validation_data()
except Exception as exc:
    st.error(f"Could not load validation data: {exc}")
    st.stop()


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔬 WAFER AI</h1>
        <p>Batch Wafer Inspection & Quality Analytics</p>
        <p>
            Automated CNN Classification |
            Defect Analytics |
            Explainable AI |
            Semiconductor Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# 6. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("⚙️ Batch Controls")
st.sidebar.success("CNN V2 Loaded")

st.sidebar.write(f"Available validation wafers: {len(X_val)}")
st.sidebar.write("Input: 64 × 64 categorical wafer maps")
st.sidebar.write(f"Output: {len(CLASS_NAMES)} defect categories")

st.sidebar.divider()

batch_size = st.sidebar.slider(
    "Number of wafers to inspect",
    min_value=10,
    max_value=len(X_val),
    value=min(100, len(X_val)),
    step=10,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1,
)

st.sidebar.caption(
    "This application evaluates the held-out validation split. "
    "The displayed dataset indices refer to rows in X_train."
)


# ============================================================
# 7. BATCH INSPECTION FUNCTION
# ============================================================

def inspect_batch(batch_X, batch_y, batch_indices):
    """
    Run CNN inference and create a result table.

    batch_indices are original row positions in the source X_train array.
    """

    batch_X = np.asarray(batch_X)
    batch_y = np.asarray(batch_y, dtype=np.int32)
    batch_indices = np.asarray(batch_indices, dtype=np.int64)

    if len(batch_X) != len(batch_y) or len(batch_y) != len(batch_indices):
        raise ValueError(
            "Images, labels and dataset indices must have equal lengths."
        )

    # Convert (N, 64, 64, 1) to (N, 64, 64).
    if batch_X.ndim == 4:
        batch_X = np.squeeze(batch_X, axis=-1)

    # Categorical values 0, 1, 2 -> one-hot channels.
    encoded = tf.one_hot(
        tf.cast(batch_X, tf.int32),
        depth=3,
        dtype=tf.float32,
    )

    probabilities = model.predict(
        encoded,
        batch_size=32,
        verbose=0,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    confidence = np.max(
        probabilities,
        axis=1,
    )

    results = pd.DataFrame(
        {
            "Dataset_Index": batch_indices,
            "Actual_Defect": [
                CLASS_NAMES[i] for i in batch_y
            ],
            "Predicted_Defect": [
                CLASS_NAMES[i] for i in predicted_labels
            ],
            "Confidence": confidence,
            "Correct": batch_y == predicted_labels,
        }
    )

    # Include probability for every class.
    for class_index, class_name in enumerate(CLASS_NAMES):
        results[f"Probability_{class_name}"] = probabilities[:, class_index]

    return results, probabilities, predicted_labels


# ============================================================
# 8. BATCH INSPECTION WORKSPACE
# ============================================================

st.subheader("Batch Inspection Workspace")
st.write(
    "Select a batch size and run CNN V2 on randomly selected "
    "validation wafers."
)

run_inspection = st.button(
    "🚀 Start Batch Inspection",
    type="primary",
    use_container_width=True,
)

if run_inspection:
    rng = np.random.default_rng(int(random_seed))

    selected_indices = rng.choice(
        len(X_val),
        size=int(batch_size),
        replace=False,
    )

    batch_X = X_val[selected_indices]
    batch_y = y_val[selected_indices]

    # Map positions within X_val back to original X_train row indices.
    original_indices = val_indices[selected_indices]

    try:
        with st.spinner(f"Inspecting {batch_size} wafers..."):
            results, probabilities, predicted_labels = inspect_batch(
                batch_X,
                batch_y,
                original_indices,
            )

            accuracy = accuracy_score(batch_y, predicted_labels)

            macro_f1 = f1_score(
                batch_y,
                predicted_labels,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )

            correct_count = int(results["Correct"].sum())
            error_count = int((~results["Correct"]).sum())

            # A model prediction of a non-"none" class is a predicted
            # defect category, not a verified physical defect count.
            predicted_defect_count = int(
                (predicted_labels != 0).sum()
            )

    except Exception as exc:
        st.error(f"Batch inspection failed: {exc}")
        st.stop()

    # ========================================================
    # 9. SAVE MISCLASSIFIED WAFERS
    # ========================================================

    error_positions = np.flatnonzero(
        batch_y != predicted_labels
    )

    error_maps = batch_X[error_positions]
    error_actual = batch_y[error_positions]
    error_predicted = predicted_labels[error_positions]
    error_indices = original_indices[error_positions]
    error_probabilities = probabilities[error_positions]

    error_npz_path = os.path.join(
        OUTPUT_DIR,
        "error_analysis.npz",
    )

    np.savez_compressed(
        error_npz_path,
        wafer_maps=error_maps,
        actual_labels=error_actual,
        predicted_labels=error_predicted,
        dataset_indices=error_indices,
        probabilities=error_probabilities,
    )

    error_results = results.loc[
        ~results["Correct"]
    ].copy()

    error_csv_path = os.path.join(
        OUTPUT_DIR,
        "misclassified_wafers.csv",
    )

    error_results.to_csv(
        error_csv_path,
        index=False,
    )

    # ========================================================
    # 10. INSPECTION SUMMARY
    # ========================================================

    st.divider()
    st.subheader("Inspection Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Wafers Inspected", len(results))
    c2.metric("Correct Predictions", correct_count)
    c3.metric("Misclassified Wafers", error_count)
    c4.metric("Batch Accuracy", f"{accuracy * 100:.2f}%")

    c5, c6 = st.columns(2)
    c5.metric("Macro F1 Score", f"{macro_f1:.4f}")
    c6.metric("Predicted Defect Categories", predicted_defect_count)

    st.caption(
        "Predicted defect counts are model outputs, not verified "
        "physical defect counts. Accuracy is measured against the "
        "validation labels."
    )

    st.success(
        f"Saved {error_count} misclassified wafer maps to "
        f"{error_npz_path}"
    )

    # ========================================================
    # 11. DEFECT DISTRIBUTION
    # ========================================================

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

    distribution = pd.DataFrame(
        {
            "Actual": actual_counts,
            "Predicted": predicted_counts,
        }
    )

    st.bar_chart(distribution)

    # ========================================================
    # 12. CONFUSION MATRIX
    # ========================================================

    st.divider()
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        batch_y,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES)),
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap = ax.imshow(
        cm,
        cmap="Blues",
    )

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Wafer Defect Confusion Matrix")

    fig.colorbar(heatmap, ax=ax)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ========================================================
    # 13. CLASSIFICATION REPORT
    # ========================================================

    st.divider()
    st.subheader("Classification Report")

    report = classification_report(
        batch_y,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    # ========================================================
    # 14. INDIVIDUAL WAFER RESULTS
    # ========================================================

    st.divider()
    st.subheader("Individual Wafer Results")

    st.dataframe(
        results,
        use_container_width=True,
        height=400,
    )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Batch Inspection CSV",
        data=csv_data,
        file_name="wafer_batch_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ========================================================
    # 15. MISCLASSIFICATION ANALYSIS
    # ========================================================

    st.divider()
    st.subheader("Misclassification Analysis")

    if error_count == 0:
        st.success(
            "No misclassified wafers were found in this batch."
        )
    else:
        st.write(
            f"CNN V2 misclassified {error_count} out of "
            f"{len(results)} wafers."
        )

        st.dataframe(
            error_results[
                [
                    "Dataset_Index",
                    "Actual_Defect",
                    "Predicted_Defect",
                    "Confidence",
                ]
            ],
            use_container_width=True,
        )

        st.download_button(
            label="⬇️ Download Misclassified Wafers CSV",
            data=error_results.to_csv(index=False).encode("utf-8"),
            file_name="misclassified_wafers.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("#### Misclassified wafer gallery")

        max_gallery_items = min(error_count, 12)
        gallery_count = st.slider(
            "Number of error examples to display",
            min_value=1,
            max_value=max_gallery_items,
            value=min(6, max_gallery_items),
        )

        for start in range(0, gallery_count, 3):
            columns = st.columns(3)

            for offset, column in enumerate(columns):
                error_position = start + offset

                if error_position >= gallery_count:
                    break

                image_index = error_positions[error_position]
                wafer_map = batch_X[image_index]

                if wafer_map.ndim == 3:
                    wafer_map = np.squeeze(wafer_map, axis=-1)

                actual_name = CLASS_NAMES[
                    int(batch_y[image_index])
                ]

                predicted_name = CLASS_NAMES[
                    int(predicted_labels[image_index])
                ]

                confidence_value = float(
                    probabilities[image_index].max()
                )

                with column:
                    fig, ax = plt.subplots(figsize=(3.2, 3.2))
                    ax.imshow(
                        wafer_map,
                        cmap="viridis",
                        interpolation="nearest",
                    )
                    ax.set_title(
                        f"Dataset row {int(original_indices[image_index])}",
                        fontsize=9,
                    )
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.markdown(
                        f"**Actual:** {actual_name}  \n"
                        f"**Predicted:** {predicted_name}  \n"
                        f"**Confidence:** {confidence_value:.2%}"
                    )

    st.success(
        f"Batch inspection completed: {len(results)} wafers analysed."
    )

    st.caption(
        "Dataset_Index is the original row position in X_train. "
        "It is not a manufacturer-provided wafer identifier."
    )

else:
    st.info(
        "Configure the batch size in the sidebar and click "
        "'Start Batch Inspection'."
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
        WAFER AI | Batch Inspection V2 |
        CNN + Computer Vision + Error Analysis
    </center>
    """,
    unsafe_allow_html=True,
)
