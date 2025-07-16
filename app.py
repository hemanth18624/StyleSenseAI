import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="StyleSense | AI Fashion AI",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for the design ---
st.markdown("""
    <style>
        /* Base and Body */
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            background-color: #F8F9FA;
        }
        .main {
            background-color: #F8F9FA;
        }
        .block-container {
            max-width: 900px; /* Centered, narrower container for vertical layout */
        }

        /* Hero Header */
        .header-container {
            background-color: #0D1B3E; /* Dark blue from Ternio-like designs */
            padding: 3rem 1rem;
            margin-bottom: 3rem;
            border-radius: 12px;
            text-align: center;
        }
        .header-container h1 {
            color: #FFFFFF;
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .header-container p {
            color: #A9B4D2; /* Lighter blue/grey for subtext */
            font-size: 1.25rem;
            font-weight: 300;
        }

        /* Content Cards */
        .content-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid #E9ECEF;
            margin-bottom: 2rem; /* Space between vertical cards */
        }
        .content-card h3 {
            font-size: 1.5rem;
            color: #0D1B3E;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #F0F2F6;
            padding-bottom: 0.5rem;
        }
        
        /* Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            color: #FFFFFF;
            background-color: #007BFF; /* Bright blue for CTA */
            border: none;
            padding: 0.75rem 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
        }
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Metric Styling */
        [data-testid="stMetric"] {
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #E9ECEF;
        }
        [data-testid="stMetric"] > div:nth-child(1) { /* Metric Label */
            color: #495057;
        }
        [data-testid="stMetric"] > div:nth-child(2) { /* Metric Value */
            color: #007BFF;
            font-size: 2rem;
            font-weight: 600;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem 1rem;
            color: #6C757D;
        }
    </style>
""", unsafe_allow_html=True)

# --- Class Names & Model Path ---
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# Model path is now set here directly
model_path = "model_weights.best.keras"


# --- Hero Header ---
st.markdown("""
    <div class="header-container">
        <h1>StyleSense AI</h1>
        <p>The Future of Fashion Classification is Here</p>
    </div>
""", unsafe_allow_html=True)


# --- Step 1: Image Upload ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("<h3>Upload An Image 📤</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop an image or click to browse.",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Garment", width=300)
else:
    st.info("Please upload an image to get started.")
st.markdown('</div>', unsafe_allow_html=True)


# --- Step 2: Prediction Results ---
if uploaded_file:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("<h3>Classification Results 📊</h3>", unsafe_allow_html=True)
    
    if st.button("✨ Analyze Fashion Item", key="predict_button"):

        # --- Model Loading and Preprocessing Functions ---
        @st.cache_resource(show_spinner="🧠 Loading AI model...")
        def load_model(path):
            try:
                return keras.models.load_model(path)
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.error("Please ensure the model file 'model_weights.best.keras' is in the correct directory.")
                return None

        def preprocess_image(pil_image):
            img = np.array(pil_image)
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            resized = cv2.resize(gray, (28, 28))
            inverted = cv2.bitwise_not(resized)
            normalized = inverted / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)
            return reshaped

        model = load_model(model_path)
        if model:
            with st.spinner("Analyzing image..."):
                preprocessed_img = preprocess_image(image)
                predictions = model.predict(preprocessed_img)[0]
                pred_idx = int(np.argmax(predictions))
                confidence = float(predictions[pred_idx])

            # Display results using st.metric
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="Predicted Category",
                    value=CLASS_NAMES[pred_idx],
                )
            with metric_col2:
                 st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.2%}",
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h5>Full Prediction Breakdown</h5>", unsafe_allow_html=True)
            df_probs = pd.DataFrame(predictions, index=CLASS_NAMES, columns=["Confidence"])
            st.bar_chart(df_probs)
    else:
        st.info("Click the button above to classify your uploaded image.")
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.markdown("---")
st.markdown("<footer>StyleSense AI © 2025 | Copyrights reversed</footer>", unsafe_allow_html=True)
