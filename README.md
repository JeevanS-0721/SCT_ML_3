# SVM Image Classification: Cats vs Dogs 🐱🐶

This project implements a Support Vector Machine (SVM) to classify images of cats and dogs. It demonstrates a fundamental machine learning pipeline: loading data, preprocessing images, flattening data for input, and training a classifier using Scikit-Learn.

This repository is part of the **SCT_ML_3** submission.

## 📂 Project Structure


## 🛠️ Technologies Used
* **Python 3.10**
* **Scikit-Image (`skimage`)**: For HOG feature extraction.
* **Scikit-Learn (`sklearn`)**: For SVM, GridSearchCV, and metrics.
* **OpenCV (`cv2`)**: For image loading and resizing.
* **NumPy**: For array manipulation.

## ⚙️ Methodology

### 1. Feature Extraction (HOG)
Instead of feeding raw pixels to the SVM (which is sensitive to lighting and color), we extract **Histogram of Oriented Gradients (HOG)**.
* **Why HOG?** It captures the *shape* and *structure* of the object (edges of ears, eyes, nose) regardless of the fur color.
* **Settings:** 9 orientations, 8x8 pixels per cell, 2x2 cells per block.

### 2. Data Preprocessing
* **Grayscale Conversion:** Reduces dimensionality.
* **Resizing:** All images resized to **64x64**.
* **Normalization:** HOG handles normalization automatically using L2-Hys block normalization.

### 3. Hyperparameter Tuning
The model uses **GridSearchCV** to automatically find the best parameters:
* **C (Regularization):** Tested `[0.1, 1, 10]`
* **Kernel:** Tested `['linear', 'rbf']`
* **Gamma:** Tested `['scale', 'auto']`

### 4. Training Configuration
* **Dataset Size:** 5,000 images (2,500 per class) were used for training to ensure robustness.
* **Split:** 80% Training, 20% Testing.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JeevanS-0721/SCT_ML_3.git](https://github.com/JeevanS-0721/SCT_ML_3.git)
    cd SCT_ML_3
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn opencv-python scikit-image matplotlib
    ```

3.  **Setup Dataset:**
    * Download the "Dogs vs Cats" dataset from Kaggle.
    * Create a folder named `dataset` in the root directory.
    * Ensure the structure is `dataset/Cat/` and `dataset/Dog/`.

4.  **Run the script:**
    ```bash
    python main.py
    ```

## 📉 Results
By using HOG features and hyperparameter tuning, the model significantly outperforms standard pixel-based SVMs.

* **Accuracy:** > 75% (Dependent on random split and grid search results)
* **Best Parameters:** Printed in the console output after GridSearch completes.

---
**Author:** JEEVAN S