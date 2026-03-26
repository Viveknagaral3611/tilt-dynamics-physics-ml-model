import time
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# ----------------------------
# System parameters
# ----------------------------
I = 250.0      # kg*m^2
m = 1000.0     # kg
h = 0.5        # m

# ----------------------------
# Time settings
# ----------------------------
t_start = 0.0
t_end = 10.0
num_points = 500
t_eval = np.linspace(t_start, t_end, num_points)

# ----------------------------
# Tilt dynamics and simulation function
# ----------------------------
def tilt_dynamics(t, x, I, m, h, c, k, v, R):
    theta, omega = x
    
    a_lat = v**2 / R
    torque = m * h * a_lat
    
    dtheta_dt = omega
    domega_dt = (torque - c * omega - k * theta) / I
    
    return [dtheta_dt, domega_dt]

def run_simulation(I, m, h, c, k, v, R):
    x0 = [0.0, 0.0]

    sol = solve_ivp(
        tilt_dynamics,
        (t_start, t_end),
        x0,
        t_eval=t_eval,
        args=(I, m, h, c, k, v, R)
    )

    theta = sol.y[0]
    omega = sol.y[1]

    max_tilt = np.max(np.abs(theta))
    final_tilt = theta[-1]
    max_omega = np.max(np.abs(omega))

    return max_tilt, final_tilt, max_omega

# ----------------------------
# Load and prepare data
# ----------------------------
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tilt_dataset.csv"))
X = df[["v", "R", "c", "k"]].values
y = df["max_tilt"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train ML model
# ----------------------------
model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation="relu",
    solver="adam",
    max_iter=3000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ----------------------------
# Compare on sample
# ----------------------------
# Pick a single test case
sample_input = X_test[0]
v_sample, R_sample, c_sample, k_sample = sample_input

# Time simulator
start_sim = time.time()
max_tilt_sim, _, _ = run_simulation(I, m, h, c_sample, k_sample, v_sample, R_sample)
end_sim = time.time()

# Time ML model
sample_scaled = scaler.transform([sample_input])

start_ml = time.time()
max_tilt_ml = model.predict(sample_scaled)[0]
end_ml = time.time()

sim_time = end_sim - start_sim
ml_time = end_ml - start_ml

print(f"Simulator output = {max_tilt_sim:.6f}")
print(f"ML output        = {max_tilt_ml:.6f}")
print(f"ML time          = {ml_time:.6f} s")

start_sim = time.time()
max_tilt_sim, _, _ = run_simulation(I, m, h, c_sample, k_sample, v_sample, R_sample)
end_sim = time.time()

print(f"Simulator time = {end_sim - start_sim:.6f} s")