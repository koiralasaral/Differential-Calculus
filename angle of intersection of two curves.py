import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables
x = sp.symbols('x')
f1 = x**2  # First curve: y = x^2
f2 = x**3  # Second curve: y = x^3

# Solve for intersection points (f1 = f2)
intersection_points = sp.solve(f1 - f2, x)

# Convert intersection points to numerical values
intersection_points = [float(point) for point in intersection_points]

# Create functions for numerical evaluation
f1_func = sp.lambdify(x, f1, 'numpy')
f2_func = sp.lambdify(x, f2, 'numpy')

# Generate x values for plotting
x_values = np.linspace(-1.5, 1.5, 500)  # Range of x values

# Evaluate the curves
y1_values = f1_func(x_values)
y2_values = f2_func(x_values)

# Plot the curves
plt.figure(figsize=(8, 6))
plt.plot(x_values, y1_values, label='y = x^2', color='blue')
plt.plot(x_values, y2_values, label='y = x^3', color='orange')

# Highlight intersection points
for point in intersection_points:
    plt.scatter(point, f1_func(point), color='red', label=f'Intersection at x = {point:.2f}')

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Two Curves')
plt.legend()
plt.grid()
plt.show()