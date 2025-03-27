import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables
x = sp.symbols('x')
f1 = x ** 3  # First curve: y = sin(x)
f2 = x ** 2  # Second curve: y = cos(x)

# Solve for intersection points (f1 = f2)
intersection_points = sp.solve(f1 - f2, x)

# Filter for real intersection points
intersection_points = [point.evalf() for point in intersection_points if point.is_real]

# Calculate the slopes at intersection points
m1 = sp.diff(f1, x)  # Derivative of f1
m2 = sp.diff(f2, x)  # Derivative of f2
slopes = [
    (float(m1.subs(x, point)), float(m2.subs(x, point)))
    for point in intersection_points
]

# Generate x values for plotting (numerically)
x_values = np.linspace(-np.pi, np.pi, 500)  # Range of x values

# Create numerical functions for evaluation
f1_func = sp.lambdify(x, f1, 'numpy')
f2_func = sp.lambdify(x, f2, 'numpy')

# Evaluate the curves
y1_values = f1_func(x_values)
y2_values = f2_func(x_values)

# Plot the curves
plt.figure(figsize=(8, 6))
plt.plot(x_values, y1_values, label='y = sin(x)', color='blue')
plt.plot(x_values, y2_values, label='y = cos(x)', color='orange')

# Highlight intersection points and plot tangents
for i, point in enumerate(intersection_points):
    # Intersection coordinates
    y_intersection = f1_func(float(point))  # Convert Sympy Float to Python float

    # Slopes of the tangents
    slope1, slope2 = slopes[i]

    # Calculate tangent lines
    tangent1 = slope1 * (x_values - float(point)) + y_intersection
    tangent2 = slope2 * (x_values - float(point)) + y_intersection

    # Plot tangents
    plt.plot(x_values, tangent1, '--', color='green', label=f'Tangent to y=sin(x) at x={point:.2f}' if i == 0 else "")
    plt.plot(x_values, tangent2, '--', color='red', label=f'Tangent to y=cos(x) at x={point:.2f}' if i == 0 else "")

    # Calculate and annotate angle of intersection
    angle = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2))) * 180 / np.pi
    plt.text(
        float(point), y_intersection, f"∠ = {angle:.2f}°", color='purple', fontsize=10, ha='center'
    )

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Two Curves with Tangents and Angle of Intersection')
plt.legend()
plt.grid()
plt.show()