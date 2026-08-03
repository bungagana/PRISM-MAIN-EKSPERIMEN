"""Minimal Streamlit demo: upload a microalgae/POME photo (or pick a bundled
sample) and see the PRISM-trained growth-phase classifier's prediction.

Run with:
    pip install streamlit
    streamlit run webapp/app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from predict import GrowthPhaseClassifier  # noqa: E402

SAMPLES_DIR = APP_DIR / "samples"
PHASE_COLOR = {"Lag": "#f2c94c", "Log": "#27ae60", "Stationary": "#2d9cdb", "Death": "#eb5757"}
PHASE_DESCRIPTION = {
    "Lag": "Adaptation phase — low chlorophyll, cells acclimating to the medium.",
    "Log": "Exponential growth — active photosynthesis, rapid biomass increase.",
    "Stationary": "Growth plateau — nutrient limitation, carotenoid accumulation.",
    "Death": "Decline phase — cell lysis, chlorophyll degradation.",
}


@st.cache_resource
def load_classifier():
    return GrowthPhaseClassifier()


def main():
    st.set_page_config(page_title="PRISM Growth-Phase Classifier", page_icon="🦠", layout="centered")
    st.title("🦠 Microalgae Growth-Phase Classifier")
    st.caption(
        "Classifies microalgae growth phase (Lag / Log / Stationary / Death) in POME "
        "medium from a photograph, using a classifier trained on PRISM-synthesized images."
    )

    try:
        clf = load_classifier()
    except FileNotFoundError:
        st.error(
            "Model not found. Run `python scripts/04_train_classifier.py` first to "
            "produce `output/classifier/models/best_growth_phase_classifier.joblib`."
        )
        return

    st.sidebar.header("Model info")
    st.sidebar.write(f"**Classifier:** {clf.metadata['best_model_name']}")
    st.sidebar.write(f"**Classes:** {', '.join(clf.phases)}")

    st.subheader("1. Choose an image")
    tab_upload, tab_sample = st.tabs(["Upload your own photo", "Try a sample image"])

    image = None
    with tab_upload:
        uploaded = st.file_uploader("Upload a JPG/PNG photo of a microalgae/POME culture", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    with tab_sample:
        sample_files = sorted(SAMPLES_DIR.glob("sample_*.jpg"))
        cols = st.columns(len(sample_files))
        chosen = None
        for col, path in zip(cols, sample_files):
            with col:
                st.image(str(path), caption=path.stem.replace("sample_", ""), use_container_width=True)
                if st.button("Use this", key=path.name):
                    chosen = path
        if chosen is not None:
            image = Image.open(chosen).convert("RGB")
            st.session_state["_chosen_sample"] = str(chosen)
        elif "_chosen_sample" in st.session_state:
            image = Image.open(st.session_state["_chosen_sample"]).convert("RGB")

    if image is None:
        st.info("Upload a photo or pick a sample above to run the classifier.")
        return

    st.subheader("2. Result")
    left, right = st.columns([1, 1.3])
    with left:
        st.image(image, caption="Input image", use_container_width=True)

    label, confidence, probs = clf.predict_array(np.asarray(image))

    with right:
        color = PHASE_COLOR.get(label, "#888")
        st.markdown(
            f"<h2 style='color:{color}'>Predicted phase: {label}</h2>",
            unsafe_allow_html=True,
        )
        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.write(PHASE_DESCRIPTION.get(label, ""))

        if probs:
            st.write("**Probability per phase:**")
            for phase in clf.phases:
                st.progress(probs[phase], text=f"{phase}: {probs[phase] * 100:.1f}%")


if __name__ == "__main__":
    main()
