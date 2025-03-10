# FashionProductRecommender
A machine learning-based Fashion Product Recommendation System that extracts features from images using deep learning and suggests visually similar products.

# Features
1. Extracts feature vectors using ResNet50 and MobileNetV2.
2. Uses K-Nearest Neighbors (KNN) for image similarity search.
3. Streamlit-based UI for uploading images and viewing recommendations.
4. Efficient caching of models and features for faster performance.
5. Supports image uploads in JPG, JPEG, PNG formats.

# Tech Stack
1. Deep Learning: TensorFlow, Keras (MobileNetV2, ResNet50)
2. Machine Learning: scikit-learn (KNN for recommendations)
3. Web Framework: Streamlit
4. Data Handling: NumPy, PIL
5. Version Control: Git LFS (for large files like images and embeddings)

# Usage
1. Upload an image in app.py to extract its feature vector.
2. Use main.py to upload an image and get visually similar product recommendations.

# Notes
1. Large files like embeddings.pkl and images use Git LFS to avoid exceeding GitHub size limits.
2. The system uses Euclidean distance to find the most similar products.
