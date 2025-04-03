import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the variables and functions
x = sp.Symbol('x')
f = x**2  # Example function f(x)
g = x + 1  # Example function g(x)

# Compute the derivatives
f_prime = sp.diff(f, x)
g_prime = sp.diff(g, x)

# Define the interval [a, b]
a, b = 1, 3

# Compute the slopes of the secant line
secant_slope = (f.subs(x, b) - f.subs(x, a)) / (g.subs(x, b) - g.subs(x, a))

# Solve for c such that f'(c)/g'(c) = secant_slope
c_value = sp.solveset(sp.Eq(f_prime / g_prime, secant_slope), x, domain=sp.Interval(a, b))

# Convert f, g, and their derivatives to NumPy functions for plotting
f_func = sp.lambdify(x, f, "numpy")
g_func = sp.lambdify(x, g, "numpy")
f_prime_func = sp.lambdify(x, f_prime, "numpy")
g_prime_func = sp.lambdify(x, g_prime, "numpy")

# Generate x values for plotting
x_vals = np.linspace(-4, 4, 400)
y_vals_f = f_func(x_vals)
y_vals_g = g_func(x_vals)

# Plot the functions and the secant lines
plt.figure(figsize=(10, 6))

# Plot f(x)
plt.plot(x_vals, y_vals_f, label=r"$f(x) = x^2$", color="blue")

# Plot g(x)
plt.plot(x_vals, y_vals_g, label=r"$g(x) = x + 1$", color="orange")  # Ensure it's plotted properly!

# Mark points a and b for f(x) and g(x)
plt.scatter([a, b], [f_func(a), f_func(b)], color="red", label=f"f endpoints: f({a}) = {f_func(a)}, f({b}) = {f_func(b)}")
plt.scatter([a, b], [g_func(a), g_func(b)], color="purple", label=f"g endpoints: g({a}) = {g_func(a)}, g({b}) = {g_func(b)}")

# Plot secant line for f(x) and g(x)
secant_line_f = secant_slope * (x_vals - a) + f_func(a)
secant_line_g = secant_slope * (x_vals - a) + g_func(a)
plt.plot(x_vals, secant_line_f, label=f"Secant Line (f)", color="cyan", linestyle="--")
plt.plot(x_vals, secant_line_g, label=f"Secant Line (g)", color="brown", linestyle="--")

# Plot tangent line(s) at c
for c in c_value:
    if c.is_real:
        c = float(c)
        tangent_slope_f = f_prime_func(c)
        tangent_slope_g = g_prime_func(c)
        tangent_f = tangent_slope_f * (x_vals - c) + f_func(c)
        tangent_g = tangent_slope_g * (x_vals - c) + g_func(c)
        plt.plot(x_vals, tangent_f, label=f"Tangent Line at c = {c:.2f} (f)", color="green", linestyle="-")
        plt.plot(x_vals, tangent_g, label=f"Tangent Line at c = {c:.2f} (g)", color="magenta", linestyle="-")
        plt.scatter([c], [f_func(c)], color="black", zorder=5, label=f"Tangent Point: c = {c:.2f}")

# Add labels and title
plt.title("Graphical Representation of Cauchy's Mean Value Theorem")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# Show the plot
plt.show()