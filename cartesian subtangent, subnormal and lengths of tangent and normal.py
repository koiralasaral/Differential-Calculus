import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the variable and the curve
x_sym = sp.Symbol('x')
y_sym = x_sym**3 

# Compute dy/dx (symbolic)
dy_dx_sym = sp.diff(y_sym, x_sym)

# Convert SymPy expressions to NumPy functions
f = sp.lambdify(x_sym, y_sym, 'numpy')
f_prime = sp.lambdify(x_sym, dy_dx_sym, 'numpy')

# Define tangent and normal equations
def tangent_eq(x, x_point, y_point, dy_dx_point):
    return dy_dx_point * (x - x_point) + y_point

def normal_eq(x, x_point, y_point, dy_dx_point):
    slope_normal = -1 / dy_dx_point  # Normal is perpendicular to tangent
    return slope_normal * (x - x_point) + y_point

# Choose a specific point (x, y)
x_point = 2  # Change this value for a different point

# Calculate the y-value at the specified x-point
y_point = f(x_point)# On the curve y = x^2

# Evaluate dy/dx at the given point
dy_dx_point = sp.lambdify(x_sym, dy_dx_sym, 'numpy')(x_point)

# Generate values for tangent and normal
x_vals_tangent_normal = np.linspace(0, 3, 100)  # Range for tangent/normal lines
tangent_vals = tangent_eq(x_vals_tangent_normal, x_point, y_point, dy_dx_point)
normal_vals = normal_eq(x_vals_tangent_normal, x_point, y_point, dy_dx_point)

# Calculate geometric properties at the given point
subtangent_val = y_point / dy_dx_point
subnormal_val = y_point * dy_dx_point
tangent_length_val = y_point * np.sqrt(1 + (1/dy_dx_point**2))
normal_length_val = y_point * np.sqrt(1 + dy_dx_point**2)

# Print the calculated values
print(f"At point ({x_point}, {y_point}):")
print(f"Subtangent: {subtangent_val}")
print(f"Subnormal: {subnormal_val}")
print(f"Length of Tangent: {tangent_length_val}")
print(f"Length of Normal: {normal_length_val}")

# Plot the curve and geometric propertie
x_vals_curve = np.linspace(-5, 5, 1000)
y_vals_curve = f(x_vals_curve)

plt.figure(figsize=(10, 10))

# Plot the curve
plt.plot(x_vals_curve, y_vals_curve, label=f'Curve: y = x^{3}', color='black', linestyle='--')

# Plot the specific point
plt.scatter(x_point, y_point, color='red', label=f'Point: ({x_point}, {y_point})')

# Plot the tangent
plt.plot(x_vals_tangent_normal, tangent_vals, label='Tangent', color='orange', linestyle='-')

# Plot the normal
plt.plot(x_vals_tangent_normal, normal_vals, label='Normal', color='purple', linestyle='-')

# Annotate subtangent and subnormal
plt.arrow(x_point, y_point, subtangent_val, 0, color='blue', head_width=0.1, label='Subtangent')
plt.arrow(x_point, y_point, 0, -subnormal_val, color='green', head_width=0.1, label='Subnormal')

# Display the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Geometric Properties of the Curve y = x^3 at a Specific Point')
plt.legend()
plt.grid()
plt.show()