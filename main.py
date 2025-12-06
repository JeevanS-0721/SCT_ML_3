import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# --- CONFIGURATION ---
DATA_DIR = './dataset'
CATEGORIES = ['Cat', 'Dog']
IMG_SIZE = 64
MAX_IMAGES = 500

data = []
labels = []

print("Loading data and extracting features... ")

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)

    count = 0
    for img_name in os.listdir(path):
        if count >= MAX_IMAGES:
            break

        try:
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # Extract HOG features
            hog_features = hog(resized_array, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

            data.append(hog_features)
            labels.append(class_num)
            count += 1
        except Exception:
            pass

print(f"Processed {len(data)} images.")

X = np.array(data)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define Hyperparameters to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print("Starting Hyperparameter Tuning... ")

# verbose=0 silences the calculation logs
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")

# Predict and Evaluate
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))
