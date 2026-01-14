# pca_digits.py
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')


# Load the data
digits = load_digits()
X = digits.data
y = digits.target

# Scale the features

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)     


# The PCA model
pca = PCA(n_components=2) # estimate only 2 PCs
X_new = pca.fit_transform(X) # project the original data into the PCA space

#Let’s plot the data before and after the PCA transform and also color code each point (sample) using the corresponding class of the digit (y) .
fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].scatter(X[:,0], X[:,1], c=y, cmap='tab10')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(X_new[:,0], X_new[:,1], c=y, cmap='tab10')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()