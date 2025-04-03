import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the Cartesian variables
x, y = sp.symbols('x y')

# Define the curve in Cartesian form
curve = x**2 + y**2 - 1  # Example: A circle x^2 + y^2 = 1

# Implicitly differentiate the curve to get dy/dx
dy_dx = sp.diff(-curve, x) / sp.diff(-curve, y)  # dy/dx = -(Fx/Fy), where Fx and Fy are partial derivatives

# Define the slope of the tangent
m = dy_dx  # Tangent slope

# Distance p from the origin to the perpendicular drawn to the tangent
p = (x * m - y) / sp.sqrt(1 + m**2)

# Display the pedal equation
pedal_equation = sp.simplify(p)
print("Pedal Equation of the Curve:")
sp.pprint(pedal_equation)

# Convert pedal equation and curve to NumPy functions for visualization
f_curve = sp.lambdify((x, y), curve, "numpy")  # Curve equation
f_pedal = sp.lambdify((x, y), pedal_equation, "numpy")  # Pedal equation

# Generate points on the curve (e.g., x^2 + y^2 = 1, circle)
theta_vals = np.linspace(0, 2 * np.pi, 300)
x_vals_curve = np.cos(theta_vals)  # Parametrize the circle
y_vals_curve = np.sin(theta_vals)

# Calculate pedal distances (p) for each (x, y) on the curve
p_vals = f_pedal(x_vals_curve, y_vals_curve)

# Plot the curve and pedal distances
plt.figure(figsize=(8, 8))

# Plot the curve
plt.plot(x_vals_curve, y_vals_curve, label="Curve: $x^2 + y^2 = 1$", color="blue")

# Plot the origin
plt.scatter(0, 0, color="red", label="Origin (0, 0)")

# Add perpendicular lines for visualization of p
for i in range(0, len(x_vals_curve), 20):  # Pick some points for illustration
    x_tangent = [0, x_vals_curve[i]]  # From origin to tangent point
    y_tangent = [0, y_vals_curve[i]]
    plt.plot(x_tangent, y_tangent, color="green", linestyle="--", alpha=0.7)

# Annotate
for i in range(0, len(x_vals_curve), 40):  # Add annotations
    plt.text(x_vals_curve[i], y_vals_curve[i], f"p={p_vals[i]:.2f}", fontsize=8, color="purple")

# Plot customization
plt.title("Pedal Equation and Visualization for $x^2 + y^2 = 1$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()