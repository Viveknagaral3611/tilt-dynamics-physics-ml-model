# Python code to model the tilt dynamics of a vehicle using a simple ODE approach
import numpy as np
import matplotlib.pyplot as plt

I = 250       # moment of inertia
m = 1000      # mass
h = 0.5       # height of COM
c = 50        # damping
k = 2000      # stiffness


v = 10        # m/s
R = 50        # meters


# Defining ODE Function
def tilt_dynamics(t, x):
    theta, omega = x
    
    a_lat = v**2 / R
    torque = m * h * a_lat
    
    dtheta_dt = omega
    domega_dt = (torque - c*omega - k*theta) / I
    
    return [dtheta_dt, domega_dt]

# Solving the ODE
from scipy.integrate import solve_ivp

t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)

x0 = [0, 0]

sol = solve_ivp(tilt_dynamics, t_span, x0, t_eval=t_eval)



# Plotting the results

# Create figure and first axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot tilt angle
ax1.plot(sol.t, sol.y[0], color='blue', linewidth=2, label='Tilt Angle')
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Tilt Angle (rad)", color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Add grid
ax1.grid(True, linestyle='--', alpha=0.6)

# Create second axis for angular velocity
ax2 = ax1.twinx()

# Plot angular velocity
ax2.plot(sol.t, sol.y[1], color='red', linewidth=2, linestyle='--', label='Angular Velocity')
ax2.set_ylabel("Angular Velocity (rad/s)", color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

# Title
plt.title("Tilt Angle and Angular Velocity Response", fontsize=14)

# Add legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Tight layout for clean spacing
plt.tight_layout()

# Show plot
plt.show()