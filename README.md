# 🐱🐶 Cat vs Dog Image Classifier using SVM

This project is a simple image classifier that uses **Support Vector Machine (SVM)** to classify images as either **Cat** or **Dog**. Built during a machine learning internship, it demonstrates the complete pipeline — from image preprocessing to model training and prediction.

---

## 📂 Project Structure

cat-dog-svm Task 3/
├── train/ # Folder containing training images (Cat/Dog)
├── test_images/ # Folder with test images for prediction
├── svm_cat_dog_model.pkl # Saved trained SVM model
├── svm_cat_dog.py # Script to train the model
├── predict_single_image.py # Predict a single image
├── predict_all_images.py # Predict all images in test_images
├── README.md # Project documentation

---

## 🔧 Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `opencv-python`
  - `scikit-learn`
  - `joblib`
  - `matplotlib` (optional)

To install dependencies:
```bash
pip install -r requirements.txt