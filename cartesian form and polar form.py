import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# 1. Cartesian Form (y = x^2)
x = sp.Symbol('x')  # Define the Cartesian variable
y = x**2  # Example curve in Cartesian form
dy_dx = sp.diff(y, x)  # Compute dy/dx
ds_dx_cartesian = sp.sqrt(1 + dy_dx**2)  # Differentiation of arc length

# Convert SymPy expression to a NumPy-compatible function
cartesian_ds_func = sp.lambdify(x, ds_dx_cartesian, 'numpy')

# Generate x values
x_vals = np.linspace(-3, 3, 200)
cartesian_ds_vals = cartesian_ds_func(x_vals)

# 2. Polar Form (r = theta^2)
theta = sp.Symbol('theta')  # Define the polar variable
r = theta**2  # Example curve in Polar form
dr_dtheta = sp.diff(r, theta)  # Compute dr/dtheta
ds_dtheta_polar = sp.sqrt(r**2 + dr_dtheta**2)  # Differentiation of arc length

# Convert SymPy expression to a NumPy-compatible function
polar_ds_func = sp.lambdify(theta, ds_dtheta_polar, 'numpy')

# Generate theta values
theta_vals = np.linspace(0, 2 * np.pi, 200)
polar_ds_vals = polar_ds_func(theta_vals)

# Plotting
plt.figure(figsize=(14, 6))

# Plot for Cartesian Form
plt.subplot(1, 2, 1)
plt.plot(x_vals, cartesian_ds_vals, label=r"$\frac{ds}{dx}$ for $y = x^2$", color='blue')
plt.title("Arc Length Differentiation (Cartesian Form)")
plt.xlabel("x")
plt.ylabel(r"$\frac{ds}{dx}$")
plt.grid()
plt.legend()

# Plot for Polar Form
plt.subplot(1, 2, 2)
plt.plot(theta_vals, polar_ds_vals, label=r"$\frac{ds}{d\theta}$ for $r = \theta^2$", color='orange')
plt.title("Arc Length Differentiation (Polar Form)")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\frac{ds}{d\theta}$")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()