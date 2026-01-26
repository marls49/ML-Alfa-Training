import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from tensorflow import keras

# 1. Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Visualize first 10 images
plt.figure(figsize=(16, 12))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.scatter(X[:, i], y, alpha=0.5)
    plt.xlabel(f'Feature {i}')
    plt.ylabel('Target')
plt.tight_layout()
plt.show()

#2. 64 Scatter plots
fig, axes = plt.subplots(8, 8, figsize=(16, 12))
for feat in range(64):
    means = [X.data[X.target == lbl, feat].mean() for lbl in range(10)]
    ax = axes[feat // 8, feat % 8]
    ax.scatter(range(10), means)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()  

# 3. Scale to [0, 1]
X_scaled = X.data / 16.0
X = X_scaled

# 4. Class counts 
class_counts = np.bincount(X.target)
print("Class counts:", class_counts)

# 5. One-hot encoder labels
encoder = OneHotEncoder()
y = encoder.fit_transform(X.target.reshape(-1, 1))

# 6. Train-test split  
X_train, X_val, y_train, y_val = train_test_split(X, y.toarray(), test_size=0.2, random_state=42)

# 7. Build the model with Keras Sequential API

model1 = keras.models.Sequential()
model1.add(keras.layers.Input(X_train.shape[1:]))
model1.add(keras.layers.Dense(300, activation="relu"))  # hidden1
model1.add(keras.layers.Dense(300, activation="relu"))  # hidden2
model1.add(keras.layers.Dense(1))  # outputlayer
model1.summary()








