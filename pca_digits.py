from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data # Die Bilddaten
y = digits.target # Die Ziffernklassen

# Standardisierung der Daten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA-Transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Visualisierung der PCA-Ergebnisse
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolor='k', s=50)
plt.xlabel('Erste Hauptkomponente')
plt.ylabel('Zweite Hauptkomponente')
plt.title('PCA der Ziffern-Daten')
plt.colorbar(scatter, label='Ziffernklasse')
plt.grid()
plt.show()