import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("tilt_dataset.csv")

# ----------------------------
# Select inputs and target
# ----------------------------
X = df[["v", "R", "c", "k"]].values
y = df["max_tilt"].values

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scale inputs
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Small neural network surrogate
# ----------------------------
model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation="relu",
    solver="adam",
    max_iter=3000,
    random_state=42
)

# ----------------------------
# Train
# ----------------------------
start_train = time.time()
model.fit(X_train_scaled, y_train)
end_train = time.time()

# ----------------------------
# Predict
# ----------------------------
start_pred = time.time()
y_pred = model.predict(X_test_scaled)
end_pred = time.time()

# ----------------------------
# Metrics
# ----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

train_time = end_train - start_train
pred_time = end_pred - start_pred

print("Training complete.")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R^2  = {r2:.6f}")
print(f"Training time   = {train_time:.6f} s")
print(f"Prediction time = {pred_time:.6f} s")


import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    linewidth=2
)
plt.xlabel("True Max Tilt (Simulator)")
plt.ylabel("Predicted Max Tilt (ML Surrogate)")
plt.title("ML Surrogate vs Physics Simulator")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

