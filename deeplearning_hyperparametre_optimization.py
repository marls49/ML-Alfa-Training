import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import warnings
import joblib

warnings.filterwarnings('ignore')  # Suppress scaler warnings

# Generate 5k realistic samples
np.random.seed(42)
n_samples = 5000
extra_X = np.random.randint(0, 10, size=(n_samples, 4)).astype(float)
extra_y = (extra_X.sum(axis=1) * 18 + np.random.normal(0, 25, n_samples)).clip(50, 1500)

df = pd.DataFrame(np.column_stack([extra_X, extra_y]), 
                  columns=['Computer', 'Monitor', 'Drucker', 'Maus', 'Budget'])

X = df[['Computer', 'Monitor', 'Drucker', 'Maus']]
y = df[['Budget']]  # DataFrame for scaler

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).ravel()

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, 
                                                    test_size=0.2, random_state=42)

# Tuned Neural Net
nn_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 20),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    batch_size=32,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_r2 = r2_score(y_test, nn_pred)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)

print(f"NN R²: {nn_r2:.4f}")
print(f"LR R²: {lr_r2:.4f}")

# Test predictions
test_cases = np.array([[1,1,1,1], [2,2,1,2], [5,4,3,4]])
test_scaled = scaler_X.transform(test_cases)
nn_test = scaler_y.inverse_transform(nn_model.predict(test_scaled).reshape(-1,1)).flatten()
lr_test = scaler_y.inverse_transform(lr_model.predict(test_scaled).reshape(-1,1)).flatten()

print("\nPredictions:")
for i, inputs in enumerate(test_cases):
    print(f"Inputs {inputs}: NN={nn_test[i]:.2f}€, LR={lr_test[i]:.2f}€")

# Save production model
joblib.dump({'nn': nn_model, 'lr': lr_model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, 
            'super_budget_model.pkl')
print("\nModel saved: super_budget_model.pkl")
