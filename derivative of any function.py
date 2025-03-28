import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variable and function
x = sp.symbols('x')

# Input the function here
function = x**3 - 4*x**2 + 6*x - 2

# Compute the derivative
derivative = sp.diff(function, x)

# Convert SymPy expressions to numerical functions
f = sp.lambdify(x, function, "numpy")
f_prime = sp.lambdify(x, derivative, "numpy")

# Generate x values
x_vals = np.linspace(-2, 5, 500)
y_vals = f(x_vals)
y_prime_vals = f_prime(x_vals)

# Plot the function and its derivative
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.plot(x_vals, y_prime_vals, label="f'(x)", color="red", linestyle="--")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

# Add titles and legend
plt.title("Function and Its Derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# Print the derivative
print("The derivative of the function is:")
print(derivative)