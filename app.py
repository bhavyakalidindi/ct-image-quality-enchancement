import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image
import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CT Image Analyzer", layout="wide")

# -----------------------------
# Custom CSS (Dashboard Style)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}

.sidebar {
    background-color: #ffffff;
}

.card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

.metric-box {
    background: #eef2ff;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

h1 {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_selected_model():
    return load_model("model/final_model.keras", compile=False)

model = load_selected_model()

# -----------------------------
# Sidebar (LIKE YOUR IMAGE)
# -----------------------------
with st.sidebar:
    st.title("🧠 CT Image Analyzer")

    option = st.selectbox(
        "Select Example",
        ["Upload Your Own", "Sample CT Scan"]
    )

    status = st.empty()

# -----------------------------
# Main Title
# -----------------------------
st.title("CT Image Analyzer – PSNR, SSIM & Model Output")

# -----------------------------
# Input Handling
# -----------------------------
if option == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload CT Image", type=["png", "jpg", "jpeg"])
else:
    uploaded_file = "sample.png"  # put a sample image in your folder

# -----------------------------
# Process Image
# -----------------------------
if uploaded_file is not None:

    if option == "Upload Your Own":
        image = Image.open(uploaded_file).convert("L")
    else:
        image = Image.open(uploaded_file).convert("L")

    image = np.array(image)

    resized = cv2.resize(image, (256, 256))
    normalized = resized / 255.0
    input_image = normalized.reshape(1, 256, 256, 1)

    with st.spinner("Analyzing CT Image..."):
        prediction = model.predict(input_image)

    enhanced = prediction[0].reshape(256, 256)
    enhanced = (enhanced * 255).astype(np.uint8)

    # -----------------------------
    # METRICS (PSNR & SSIM)
    # -----------------------------
    psnr_value = peak_signal_noise_ratio(resized, enhanced)
    ssim_value = structural_similarity(resized, enhanced)

    status.success("✅ Analysis Complete")

    # -----------------------------
    # IMAGE DISPLAY (LIKE CARDS)
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Enchanced Image")
        st.image(resized, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(enhanced, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # METRICS DISPLAY
    # -----------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.subheader("PSNR")
        st.write(f"{psnr_value:.2f} dB")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.subheader("SSIM")
        st.write(f"{ssim_value:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------
    _, buffer = cv2.imencode(".png", enhanced)
    byte_img = io.BytesIO(buffer)

    st.download_button(
        label="📥 Download Enhanced Image",
        data=byte_img,
        file_name="enhanced.png",
        mime="image/png"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
