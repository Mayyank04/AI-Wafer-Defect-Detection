
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Wafer CNN Error Investigator",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Wafer CNN Error Investigator")
st.caption("CNN V2 | Grad-CAM Misclassification Analysis")

CSV_PATH = Path("outputs/gradcam_errors/error_summary.csv")
IMAGE_DIR = Path("outputs/gradcam_errors")

if not CSV_PATH.exists():
    st.error("Error summary not found. Run gradcam_errors.py first.")
    st.stop()

df = pd.read_csv(CSV_PATH)

st.subheader("Error Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Misclassified Wafers", len(df))
col2.metric(
    "High-Confidence Errors",
    int((df["Confidence"] >= 0.90).sum())
)
col3.metric(
    "Average Error Confidence",
    f"{df['Confidence'].mean():.1%}"
)

st.divider()

st.subheader("Most Common Misclassifications")

confusions = (
    df.groupby(["Actual_Defect", "Predicted_Defect"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

st.dataframe(confusions, use_container_width=True)

st.divider()

st.subheader("Investigate Individual Wafers")

selected_index = st.selectbox(
    "Select wafer dataset index",
    df["Dataset_Index"].tolist()
)

selected = df.loc[
    df["Dataset_Index"] == selected_index
].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Actual defect:** {selected['Actual_Defect']}")
    st.write(f"**Predicted defect:** {selected['Predicted_Defect']}")
    st.write(f"**Prediction confidence:** {selected['Confidence']:.2%}")
    st.write(
        f"**Actual-class probability:** "
        f"{selected['Actual_Class_Probability']:.2%}"
    )

with col2:
    if selected["Confidence"] >= 0.90:
        st.error("High-confidence misclassification")
    else:
        st.warning("Misclassification")

image_path = IMAGE_DIR / str(selected["GradCAM_Image"])

if image_path.exists():
    st.image(
        str(image_path),
        caption=f"Grad-CAM investigation: wafer {selected_index}",
        use_container_width=True
    )
else:
    st.warning(f"Image not found: {image_path}")

st.divider()

st.subheader("Confidence Distribution")
st.bar_chart(
    df["Confidence"].mul(100).round(1).value_counts(
        bins=10
    ).sort_index()
)

st.download_button(
    "Download Error Summary CSV",
    data=df.to_csv(index=False),
    file_name="wafer_error_summary.csv",
    mime="text/csv"
)
