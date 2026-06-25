import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .pneumonia {
        background-color: #ffe4e4;
        border: 2px solid #e53e3e;
        color: #c53030;
    }
    .normal {
        background-color: #e6ffed;
        border: 2px solid #38a169;
        color: #276749;
    }
    .disclaimer {
        background-color: #fff8e1;
        border-left: 4px solid #f6ad55;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #744210;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
THRESHOLD = 0.7
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_densenet_pneumonia.keras")
    return model

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🫁 Chest X-Ray Pneumonia Detector")
st.markdown("Upload a chest X-ray image and the AI model will predict whether it shows **Normal** lungs or signs of **Pneumonia**.")

st.markdown("---")

# ── Sidebar: Model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About the Model")
    st.markdown("""
    - **Architecture:** DenseNet121
    - **Task:** Binary Classification  
    - **Test Accuracy:** 91%
    - **Input Size:** 224 × 224 px
    - **Classes:** Normal · Pneumonia
    - **Dataset:** Chest X-Ray Images (Kaggle)
    - **Framework:** TensorFlow / Keras
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "91%")
        st.metric("Precision", "~92%")
    with col2:
        st.metric("Recall", "~93%")
        st.metric("F1 Score", "~92%")

    st.markdown("---")
    st.caption("Built by **Nupur Zile** · DenseNet121 Transfer Learning")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray (PA or AP view) in JPG or PNG format."
)

if uploaded_file is not None:

    # Load & display image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array  = np.array(img_resized, dtype=np.float32)
    img_array  = np.expand_dims(img_array, axis=0)
    img_array  = preprocess_input(img_array)

    # Predict
    with st.spinner("🔍 Analyzing X-ray..."):
        try:
            model = load_model()
            prediction = model.predict(img_array, verbose=0)[0][0]
        except Exception as e:
            st.error(f"⚠️ Model not found: `best_densenet_pneumonia.keras`\n\n{e}")
            st.info("Make sure the trained `.keras` model file is in the same folder as `app.py`.")
            st.stop()

    # Results
    label      = "PNEUMONIA" if prediction >= THRESHOLD else "NORMAL"
    confidence = prediction if label == "PNEUMONIA" else (1 - prediction)
    conf_pct   = round(confidence * 100, 2)

    st.markdown("### 🩺 Prediction Result")

    if label == "PNEUMONIA":
        st.markdown(f"""
        <div class="result-box pneumonia">
            🔴 Pneumonia Detected — {conf_pct}% confidence
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box normal">
            🟢 Normal — {conf_pct}% confidence
        </div>""", unsafe_allow_html=True)

    # Confidence bar
    st.markdown("#### Confidence Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Normal**")
        st.progress(float(1 - prediction))
        st.caption(f"{round((1-prediction)*100, 2)}%")
    with col2:
        st.markdown("**Pneumonia**")
        st.progress(float(prediction))
        st.caption(f"{round(prediction*100, 2)}%")

    # Raw score
    with st.expander("🔢 Raw Prediction Score"):
        st.write(f"Sigmoid output: `{round(float(prediction), 4)}`")
        st.write(f"Decision threshold: `{THRESHOLD}`")
        st.write("Score ≥ 0.70 → Pneumonia | Score < 0.70 → Normal")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is built for educational 
        and portfolio purposes only. It is <strong>NOT</strong> a substitute for professional 
        medical diagnosis. Always consult a qualified radiologist or physician for 
        clinical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder UI
    st.info("👆 Upload a chest X-ray image above to get started.")
    with st.expander("ℹ️ What kind of images work best?"):
        st.markdown("""
        - Frontal chest X-rays (PA view preferred)
        - JPEG or PNG format
        - Clear, well-exposed radiographs
        - Images from the chest X-ray dataset (Kaggle) work best
        """)
