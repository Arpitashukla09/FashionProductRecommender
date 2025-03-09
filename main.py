import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# ✅ Load saved features & filenames (Using memory mapping for faster access)
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")), dtype=np.float32)
filenames = pickle.load(open("filenames.pkl", "rb"))

# ✅ Preload and cache ResNet50 model
@st.cache_resource
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

model = load_model()

# ✅ Cache Nearest Neighbors model for faster recommendations
@st.cache_resource
def load_knn():
    neighbors = NearestNeighbors(n_neighbors=6, metric="euclidean", algorithm="kd_tree")
    neighbors.fit(feature_list)
    return neighbors

knn_model = load_knn()

st.title("🛍️ Fashion Product Recommender")

def save_uploaded_file(uploaded_file):
    """Saves uploaded file efficiently."""
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_features(img_path, model):
    """Optimized feature extraction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    img_array = preprocess_input(img_array)  # Normalize input
    
    # 🔥 Optimized direct inference
    features = model(img_array, training=False).numpy().flatten()
    return features / np.linalg.norm(features)  # Normalize for accuracy

def recommend(features, knn_model):
    """Finds similar images using pre-fitted Nearest Neighbors model."""
    distances, indices = knn_model.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("📤 Upload an image")

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    # 🔹 Creating side-by-side layout
    col1, col2 = st.columns([1, 2])  # col1 (uploaded image) & col2 (recommendations)
    
    with col1:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)  

    with col2:
        st.subheader("🔹Recommended Products:")
        rec_cols = st.columns(5)
        features = extract_features(file_path, model)
        indices = recommend(features, knn_model)

        for i in range(5):
            with rec_cols[i]:
                st.image(filenames[indices[0][i + 1]], use_container_width=True)  # ✅ Fixed Warning