An AI-driven fashion product recommendation system that uses ResNet50 for feature extraction and K-Nearest Neighbors (KD-Tree) for image similarity search. This web app allows users to upload an image and get visually similar product recommendations in real-time.

📂 Dataset:  (https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

# 🚀 Features
✅ Deep Learning for Feature Extraction using ResNet50
✅ Content-Based Image Retrieval with KNN (KD-Tree)
✅ Efficient Feature Processing with GlobalMaxPooling2D & preprocess_input
✅ Real-time Image Upload & Recommendation UI with Streamlit
✅ Optimized for Performance using memory-mapped data structures
✅ Supports Various Image Formats including JPG, PNG

# 📌 Tech Stack
Programming Language: Python
Machine Learning & Deep Learning: TensorFlow, Keras, ResNet50, MobileNetV2
Data Processing & Storage: NumPy, Pickle, PIL (Pillow), Base64
Image Search Algorithm: Scikit-learn (KNN with KD-Tree)
Web Framework: Streamlit

# 📸 How It Works?
1️⃣ Upload an image using the Streamlit UI
2️⃣ The system extracts deep features using ResNet50
3️⃣ A KNN model (KD-Tree) finds the most similar products
4️⃣ Top-5 recommended products are displayed

# 📜 File Structure

📂 fashion-recommender
 ┣ 📂 data/
 ┃ ┣ 📂 images/           # Dataset images
 ┃📜 embeddings.pkl    # Precomputed feature embeddings
 ┃📜 filenames.pkl     # Image filenames list
 ┣ 📂 uploads/            # Uploaded images (runtime)
 ┣ 📜 main.py             # Streamlit app UI & recommendation logic
 ┣ 📜 app.py
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 feature_extraction.py # Script to extract features (if needed)
 ┣ 📜 README.md            # Project documentation


# 📝 License
This project is open-source and available under the MIT License.
