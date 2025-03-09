import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import MobileNetV2, preprocess_input
import numpy as np
import pickle
import base64
from PIL import Image

# Set page config
st.set_page_config(page_title="Fashion Feature Extractor", layout="wide", page_icon="🛍️")

# Custom CSS for Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        .title {
            font-size: 34px;
            font-weight: bold;
            color: #ff6600;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            color: #666;
            text-align: center;
        }
        .info-box {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #999;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<p class='title'>🔍 Fashion Feature Extractor</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload an image to extract features using MobileNetV2.</p>", unsafe_allow_html=True)
st.write("---")

# Load MobileNetV2 model
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.Sequential([model, GlobalMaxPooling2D()])

model = load_model()

# Feature extraction function
def extract_features(img_file, model):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array).flatten()
    return features / np.linalg.norm(features)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<div class='info-box'>✨ Extracting Features...</div>", unsafe_allow_html=True)
        
        # Extract features
        with st.spinner("🔄 Processing... Please wait"):
            features = extract_features(uploaded_file, model)
        
        st.success("✅ Feature Extraction Complete!")
        st.write("### Extracted Feature Vector:")
        st.text(features[:10])  # Display first 10 feature values
        
        # Feature Download Option
        if st.button("💾 Save Features"):
            file_name = "features.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(features, f)
            
            with open(file_name, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()

            href = f'<a href="data:file/pkl;base64,{b64}" download="{file_name}">📥 Download Features</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Features ready for download!")

st.write("---")
st.markdown("<p class='footer'>Developed by Arpita Shukla | Powered by TensorFlow & Streamlit</p>", unsafe_allow_html=True)
