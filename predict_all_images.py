import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Load trained SVM model
model = joblib.load("/Users/prashantyadv/Prodigy Intership/cat-dog-svm Task 3/svm_cat_dog_model.pkl")

# Folder containing test images
test_folder = "/Users/prashantyadv/Prodigy Intership/cat-dog-svm Task 3/test_images"
image_paths = glob(os.path.join(test_folder, "*.*"))  # all files

# Supported formats
valid_exts = (".jpg", ".jpeg", ".png", ".jfif", ".webp", ".avif")

for path in image_paths:
    if not path.lower().endswith(valid_exts):
        continue  # skip non-image files

    # Load and preprocess image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load {path}")
        continue

    img_resized = cv2.resize(img, (64, 64))
    img_flattened = img_resized.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_flattened)
    label = "Dog 🐶" if prediction[0] == 1 else "Cat 🐱"

    # Show image
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"{os.path.basename(path)}\nPrediction: {label}")
    plt.axis('off')
    plt.show()

    print(f"📷 {os.path.basename(path)} => {label}")

