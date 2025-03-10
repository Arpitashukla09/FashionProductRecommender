# FashionProductRecommender
A machine learning-based Fashion Product Recommendation System that extracts features from images using deep learning and suggests visually similar products.

# Features
1. Extracts feature vectors using ResNet50 and MobileNetV2.
2. Uses K-Nearest Neighbors (KNN) for image similarity search.
3. Streamlit-based UI for uploading images and viewing recommendations.
4. Efficient caching of models and features for faster performance.
5. Supports image uploads in JPG, JPEG, PNG formats.

# Tech Stack
Deep Learning: TensorFlow, Keras (MobileNetV2, ResNet50)
Machine Learning: scikit-learn (KNN for recommendations)
Web Framework: Streamlit
Data Handling: NumPy, PIL
Version Control: Git LFS (for large files like images and embeddings)

# Usage
Upload an image in app.py to extract its feature vector.
Use main.py to upload an image and get visually similar product recommendations.

# Notes
Large files like embeddings.pkl and images use Git LFS to avoid exceeding GitHub size limits.
The system uses Euclidean distance to find the most similar products.
