import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the polar variables
theta = sp.Symbol('theta')
r = theta**2  # Example polar curve: r = theta^2

# Compute dr/dtheta
dr_dtheta = sp.diff(r, theta)

# Define the pedal equation in polar form
p_polar = r**2 / sp.sqrt(r**2 + dr_dtheta**2)

# Simplify the pedal equation
p_polar_simplified = sp.simplify(p_polar)
print("Pedal Equation in Polar Form:")
sp.pprint(p_polar_simplified)

# Convert pedal equation to a NumPy-compatible function for visualization
f_p_polar = sp.lambdify(theta, p_polar_simplified, "numpy")

# Generate theta values, avoiding theta = 0
theta_vals = np.linspace(0.01, 2 * np.pi, 300)  # Start from a small positive value

# Calculate pedal distances
p_vals_polar = f_p_polar(theta_vals)

# Plot the pedal equation for the polar curve
plt.figure(figsize=(8, 6))

plt.plot(theta_vals, p_vals_polar, label="Pedal Distance $p$", color="blue")
plt.title("Pedal Equation in Polar Form for $r = \\theta^2$")
plt.xlabel(r"$\theta$")
plt.ylabel("p")
plt.grid(True)
plt.legend()
plt.show()