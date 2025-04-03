import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the variable and function
x = sp.Symbol('x')
f = x**3 + 2*x  # Example function

# Derivative of the function
f_prime = sp.diff(f, x)

# Define the interval [a, b]
a, b = -5, 5

# Find the slope of the secant line
secant_slope = (f.subs(x, b) - f.subs(x, a)) / (b - a)

# Solve for c such that f'(c) = secant_slope
c_value = sp.solveset(sp.Eq(f_prime, secant_slope), x, domain=sp.Interval(a, b))

# Convert f and f_prime to NumPy functions for plotting
f_func = sp.lambdify(x, f, "numpy")
f_prime_func = sp.lambdify(x, f_prime, "numpy")

# Generate x values for plotting
x_vals = np.linspace(-10, 10, 400)
y_vals = f_func(x_vals)

# Plot the function and the secant line
plt.figure(figsize=(10, 6))

# Plot the curve
plt.plot(x_vals, y_vals, label=r"$f(x) = x^3 + 2x$", color="blue")

# Plot the secant line
secant_line = secant_slope * (x_vals - a) + f_func(a)
plt.plot(x_vals, secant_line, label=f"Secant Line (Slope = {secant_slope:.2f})", color="orange", linestyle="--")

# Plot tangent lines at c
for c in c_value:
    if c.is_real:
        y_c = f_func(float(c))
        tangent_line = f_prime_func(float(c)) * (x_vals - float(c)) + y_c
        plt.plot(x_vals, tangent_line, label=f"Tangent Line at c = {float(c):.2f}", color="green", linestyle="-")
        plt.scatter([float(c)], [y_c], color="red", zorder=5, label=f"Tangent Point: c = {float(c):.2f}")

# Mark points a and b
plt.scatter([a, b], [f_func(a), f_func(b)], color="purple", zorder=5, label=f"Endpoints: a = {a}, b = {b}")

# Add labels and title
plt.title("Graphical Representation of Lagrange's Mean Value Theorem")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show the plot
plt.show()