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

# Calculate the slopes at intersection points
m1 = sp.diff(f1, x)  # Derivative of f1
m2 = sp.diff(f2, x)  # Derivative of f2
slopes = [
    (float(m1.subs(x, point)), float(m2.subs(x, point)))
    for point in intersection_points
]

# Generate numerical functions for the curves
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

# Highlight intersection points and plot tangents
for i, point in enumerate(intersection_points):
    # Intersection coordinates
    y_intersection = f1_func(point)

    # Slopes of the tangents
    slope1, slope2 = slopes[i]

    # Calculate tangent lines
    tangent1 = slope1 * (x_values - point) + y_intersection
    tangent2 = slope2 * (x_values - point) + y_intersection

    # Plot tangents
    plt.plot(x_values, tangent1, '--', color='green', label=f'Tangent to y=x^2 at x={point:.2f}' if i == 0 else "")
    plt.plot(x_values, tangent2, '--', color='red', label=f'Tangent to y=x^3 at x={point:.2f}' if i == 0 else "")

    # Calculate and annotate angle of intersection
    angle = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2))) * 180 / np.pi
    plt.text(
        point, y_intersection, f"∠ = {angle:.2f}°", color='purple', fontsize=10, ha='center'
    )

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Two Curves with Tangents and Angle of Intersection')
plt.legend()
plt.grid()
plt.show()