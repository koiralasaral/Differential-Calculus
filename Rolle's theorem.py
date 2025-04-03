import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variable and the function
x = sp.Symbol('x')
f = x**3 - 3*x + 2  # Example function

# Find the derivative
f_prime = sp.diff(f, x)

# Find critical points (where f'(x) = 0)
critical_points = sp.solveset(f_prime, x, domain=sp.S.Reals)

# Evaluate function at the boundaries a and b
a, b = -2, 2  # Closed interval [a, b]
f_a = f.subs(x, a)
f_b = f.subs(x, b)

# Check Rolle's theorem conditions
print(f"f(a) = f({a}) = {f_a}")
print(f"f(b) = f({b}) = {f_b}")
if f_a == f_b:
    print("Rolle's theorem applies: f(a) = f(b)")

# Convert f and f_prime to NumPy functions for plotting
f_func = sp.lambdify(x, f, "numpy")
f_prime_func = sp.lambdify(x, f_prime, "numpy")

# Generate x values for plotting
x_vals = np.linspace(a - 1, b + 1, 400)  # Slightly extend the range
y_vals = f_func(x_vals)
y_prime_vals = f_prime_func(x_vals)

# Plot the function and its derivative
plt.figure(figsize=(10, 6))

# Plot the function f(x)
plt.plot(x_vals, y_vals, label=r"$f(x) = x^3 - 3x + 2$", color="blue")
plt.axhline(0, color='black', linewidth=0.8, linestyle="--")  # x-axis

# Mark points on the graph
plt.scatter([a, b], [f_a, f_b], color="red", label=f"Endpoints: f({a}) = f({b}) = {f_a}")

# Plot critical points where f'(c) = 0
for cp in critical_points:
    if cp.is_real and a < cp.evalf() < b:  # Check if the critical point is in the interval
        y_cp = f.subs(x, cp)
        plt.scatter(float(cp), float(y_cp), color="green", label=f"Critical point: c = {cp.evalf():.2f}")
        plt.annotate(f"c = {cp.evalf():.2f}", (float(cp), float(y_cp)), textcoords="offset points", xytext=(10, -15), arrowprops=dict(arrowstyle="->", color="green"))

# Add labels and title
plt.title("Graphical Representation of Rolle's Theorem")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show the plot
plt.show()