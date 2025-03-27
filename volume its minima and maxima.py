
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables
h, r = sp.symbols('h r', positive=True)

# Radius R in terms of h and r (geometrical relation)
R = (h - r) * sp.tan(sp.asin(r / (h - r)))

# Volume of the cone
V = (1/3) * sp.pi * R**2 * h

# Define r's numerical value for plotting
r_value = 1  # Example: set r = 1

# Generate numerical function for volume using lambdify
V_func = sp.lambdify(h, V.subs(r, r_value), 'numpy')

# Plot the volume as a function of h
h_values = np.linspace(2*r_value, 6*r_value, 500)  # Range of h values
V_values = V_func(h_values)  # Evaluate the volume at h_values

plt.figure(figsize=(8, 6))
plt.plot(h_values, V_values, label='Volume of Cone')
plt.axvline(h_values[np.argmin(V_values)], color='red', linestyle='--', label='Minimum Volume')
plt.xlabel('Altitude (h)')
plt.ylabel('Volume (V)')
plt.title('Volume of Cone vs Altitude')
plt.legend()
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
r = 1  # Radius of the sphere
h = 4 * r  # Altitude of the cone (from the problem's solution)
theta = np.arcsin(1 / 3)  # Semi-vertical angle of the cone

# Generate points for the sphere
phi, psi = np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50)
phi, psi = np.meshgrid(phi, psi)
x_sphere = r * np.sin(phi) * np.cos(psi)
y_sphere = r * np.sin(phi) * np.sin(psi)
z_sphere = r * np.cos(phi)

# Generate points for the cone
z_cone = np.linspace(0, h, 50)  # Height values of the cone
radius_cone = z_cone / np.tan(theta)
x_cone = np.outer(radius_cone, np.cos(psi[0]))
y_cone = np.outer(radius_cone, np.sin(psi[0]))
z_cone = np.outer(z_cone, np.ones_like(psi[0]))

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
ax.plot_surface(x_sphere, y_sphere, z_sphere + r, color='blue', alpha=0.5, label="Sphere")

# Plot the cone
ax.plot_surface(x_cone, y_cone, z_cone, color='black', alpha=0.9, label="Cone")

# Set plot limits and labels
ax.set_xlim([-2 * r, 2 * r])
ax.set_ylim([-2 * r, 2 * r])
ax.set_zlim([0, h + r])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Cone Circumscribed around a Sphere')

plt.show()