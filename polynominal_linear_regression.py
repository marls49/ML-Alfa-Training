import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns; sns.set()

# Create sample data
rng = np.random.default_rng(1)
x = 10 * rng.random(50)
y = 2 * x - 5 + rng.normal(0, 10, 50)

# Preppare matrix with polynomial
poly = PolynomialFeatures(3, include_bias=False)
X_train = poly.fit_transform(x[:, None])

# Fit linear regression on polynomial features
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit_transform (X_train, y)                                    

# Generate predictions
x_fit = np.linspace(0, 10, 100)
X_fit_poly = poly.transform(x_fit[:, None])
y_fit = model.predict(X_fit_poly)           

# Plot results
plt.scatter(x, y, label='Data Points')
plt.plot(x_fit, y_fit)
plt.show()


