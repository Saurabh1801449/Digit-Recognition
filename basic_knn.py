import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Load the image safely
image_path = 'images/digits.png'

# Debugging: Print absolute path
abs_path = os.path.abspath(image_path)
print(f"Trying to load image from: {abs_path}")

data = cv2.imread(image_path)

if data is None:
    print(f"❌ Error: Could not load image at '{image_path}'. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Resize to a fixed size (ensure it's divisible by 50 & 100)
height, width = gray.shape
new_height = (height // 50) * 50
new_width = (width // 100) * 100
gray = cv2.resize(gray, (new_width, new_height))

# Use Matplotlib instead of cv2.imshow()
plt.imshow(gray, cmap='gray')
plt.title("Processed Image")
plt.axis('off')
plt.show()

# Splitting the image into 50 rows and 100 columns
arr = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
arr = np.array(arr)
print("✅ Resulting Shape:", arr.shape)  # Should be (50, 100, 20, 20)

# Flatten images for training/testing
X_train = arr[:, :70].reshape(-1, 400).astype(np.float32)
X_test = arr[:, 70:100].reshape(-1, 400).astype(np.float32)
print("🔹 Input Shapes\n--> Train: {}, Test: {}".format(X_train.shape, X_test.shape))

# Define labels (Assuming 10 unique classes)
y = np.arange(10)

y_train = np.repeat(y, X_train.shape[0] // 10)[:, np.newaxis]
y_test = np.repeat(y, X_test.shape[0] // 10)[:, np.newaxis]
print("🔹 Target Shapes\n--> Train: {}, Test: {}".format(y_train.shape, y_test.shape))

# Train a KNN classifier
classifier_knn = cv2.ml.KNearest_create()
classifier_knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
response, result, neighbours, distance = classifier_knn.findNearest(X_test, k=3)

# Calculate accuracy
correct = result == y_test
correct = np.count_nonzero(correct)
accuracy = correct * (100.0 / result.size)
print("✅ Accuracy:", accuracy)
