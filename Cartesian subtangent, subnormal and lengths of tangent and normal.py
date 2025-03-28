import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variable and function
x = sp.symbols('x')
function = x**3 - 3*x**2 + 4  # Change this to any function you like

# Compute the derivative
derivative = sp.diff(function, x)

# Convert SymPy expressions to numerical functions
f = sp.lambdify(x, function, "numpy")
f_prime = sp.lambdify(x, derivative, "numpy")

# Define the x-point for tangency
x_point = 5  # Change this to the desired point of analysis
y_point = f(x_point)
slope = f_prime(x_point)
# Check if slope is zero
if slope == 0:
    print(f"The slope at x = {x_point} is zero, meaning the tangent is horizontal.")
    subtangent = None
    subnormal = None
    tangent_length = None
    normal_length = None
else:
    # Calculate subtangent, subnormal, and lengths
    subtangent = y_point / slope
    subnormal = -y_point / slope
    tangent_length = np.sqrt(1 + slope**2) * abs(subtangent)
    normal_length = np.sqrt(1 + slope**2) * abs(subnormal)
    
    print("Subtangent:", subtangent)
    print("Subnormal:", subnormal)
    print("Length of Tangent:", tangent_length)
    print("Length of Normal:", normal_length)
    # Generate x and y values for plotting
    x_vals = np.linspace(-10, 10, 500)
    y_vals = f(x_vals)
    tangent = slope * (x_vals - x_point) + y_point
    normal = -(1 / slope) * (x_vals - x_point) + y_point
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.plot(x_vals, tangent, label="Tangent", color="green", linestyle="--")
    plt.plot(x_vals, normal, label="Normal", color="orange", linestyle="--")
    plt.scatter([x_point], [y_point], color="red", label=f"Point of Tangency ({x_point}, {y_point})")

# Add titles and legend
    plt.title("Subtangent, Subnormal, Tangent, and Normal Visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid(alpha=0.3)

plt.show()




