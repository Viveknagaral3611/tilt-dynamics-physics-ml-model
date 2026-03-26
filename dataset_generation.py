import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ----------------------------
# Fixed system parameters
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
# Tilt dynamics function
# ----------------------------
def tilt_dynamics(t, x, I, m, h, c, k, v, R):
    theta, omega = x
    
    a_lat = v**2 / R
    torque = m * h * a_lat
    
    dtheta_dt = omega
    domega_dt = (torque - c * omega - k * theta) / I
    
    return [dtheta_dt, domega_dt]

# ----------------------------
# Helper: simulate one case
# ----------------------------
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

    # Output features from simulator
    max_tilt = np.max(np.abs(theta))
    final_tilt = theta[-1]
    max_omega = np.max(np.abs(omega))

    return max_tilt, final_tilt, max_omega

# ----------------------------
# Parameter sweep ranges
# ----------------------------
v_values = np.linspace(5, 25, 8)        # m/s
R_values = np.linspace(20, 120, 8)      # m
c_values = np.linspace(30, 120, 5)      # damping
k_values = np.linspace(1500, 3500, 5)   # stiffness

# ----------------------------
# Generate dataset
# ----------------------------
data = []

for v in v_values:
    for R in R_values:
        for c in c_values:
            for k in k_values:
                max_tilt, final_tilt, max_omega = run_simulation(I, m, h, c, k, v, R)
                
                data.append({
                    "v": v,
                    "R": R,
                    "c": c,
                    "k": k,
                    "max_tilt": max_tilt,
                    "final_tilt": final_tilt,
                    "max_omega": max_omega
                })

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("tilt_dataset.csv", index=False)

print("Dataset saved as tilt_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())