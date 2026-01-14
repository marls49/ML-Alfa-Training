# culminative variance (np.cumsum)

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
plt.style.use('ggplot')     

# Load the data
digits = load_digits()
X = digits.data
y = digits.target

# Culminative variance plot 
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Culminative Variance Plot')
plt.show()

