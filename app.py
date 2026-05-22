import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set up page config
st.set_page_config(page_title="Oral Health Assistant", layout="centered")

# Medical Disclaimer at the top
st.warning("⚠️ **Medical Disclaimer:** This tool is an AI prototype for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.")

# 1. Load your trained model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("oral_cancer_model.keras")

try:
    model = load_my_model()
    st.success("AI Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Build the website interface
st.title("Oral Health Classification Assistant")
st.write("Upload a close-up image of the oral lesion to check the AI prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Analyzing image...")

    # 3. Preprocess the image 
    img = image.resize((224, 224)) 
    img_array = np.array(img) / 255.0  # Normalize pixels
    img_array = np.expand_dims(img_array, axis=0) # Reshape for model batching

    # 4. Make prediction
    prediction = model.predict(img_array)
    raw_val = prediction[0][0]
    
    st.write(f"**Raw Prediction Value:** {raw_val:.6f}")
    
    # Threshold logic
    if raw_val > 0.5: 
        st.balloons()
        st.success(f"**Prediction:** Healthy\n\n**Confidence:** {raw_val * 100:.2f}%")
    else:
        st.error(f"**Prediction:** Action Required / Unhealthy\n\n**Confidence:** {(1 - raw_val) * 100:.2f}%")
