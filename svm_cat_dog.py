import os
import cv2
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # For MacOS GUI support

# ✅ Path to your dataset
DATA_DIR = "/Users/prashantyadv/Prodigy Intership/cat-dog-svm Task 3/train"

# ✅ Load and preprocess images
def load_images(folder, size=(64, 64), max_images_per_class=2000):
    data = []
    cat_count = 0
    dog_count = 0

    for filename in os.listdir(folder):
        if not filename.endswith(".jpg"):
            continue

        label = 0 if "cat" in filename.lower() else 1
        if label == 0 and cat_count >= max_images_per_class:
            continue
        if label == 1 and dog_count >= max_images_per_class:
            continue

        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, size)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data.append((img_gray.flatten(), label))

        if label == 0:
            cat_count += 1
        else:
            dog_count += 1

        if cat_count >= max_images_per_class and dog_count >= max_images_per_class:
            break

    print(f"✅ Loaded {len(data)} images: {cat_count} cats, {dog_count} dogs")
    return data

# ✅ Load Data
all_data = load_images(DATA_DIR, max_images_per_class=2000)  # Use 2k per class for faster training

# ✅ Shuffle and split
random.shuffle(all_data)
X = [item[0] for item in all_data]
y = [item[1] for item in all_data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train SVM model
print("🔁 Training SVM...")
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print("✅ Training complete.")

# ✅ Evaluate
y_pred = clf.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Show 3 random predictions
for i in range(3):
    index = random.randint(0, len(X_test) - 1)
    img = np.array(X_test[index]).reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {'Dog' if y_pred[index] == 1 else 'Cat'} | Actual: {'Dog' if y_test[index] == 1 else 'Cat'}")
    plt.axis('off')
    plt.show()
# ✅ Visualize some correct predictions where Predicted == Actual == Cat
shown = 0
for index in range(len(X_test)):
    if y_pred[index] == 0 and y_test[index] == 0:
        img = np.array(X_test[index]).reshape(64, 64)
        plt.imshow(img, cmap='gray')
        plt.title("✅ Predicted: Cat | ✅ Actual: Cat")
        plt.axis('off')
        plt.show()
        shown += 1
    if shown == 5:
        break

# ✅ Save the model
import joblib
joblib.dump(clf, "svm_cat_dog_model.pkl")
print("✅ Model saved as svm_cat_dog_model.pkl")
