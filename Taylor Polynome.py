import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the function and variables
x, c = sp.symbols('x c')
f = sp.sin(x)  # Function f(x) = e^x
a = 0  # Expansion point

# Function to calculate the Taylor polynomial and remainder
def taylor_polynomial(f, a, n):
    taylor = sum(sp.diff(f, x, i).subs(x, a) * (x - a)**i / sp.factorial(i) for i in range(n + 1))
    remainder = sp.diff(f, x, n + 1).subs(x, c) * (x - a)**(n + 1) / sp.factorial(n + 1)
    return taylor, remainder

# Print and plot Taylor polynomials for n = 1, 2, 3, 4
n_values = [1, 2, 3, 4]
x_vals = np.linspace(-10, 10, 400)
f_func = sp.lambdify(x, f, 'numpy')  # Convert f(x) to a NumPy function

plt.figure(figsize=(12, 8))
plt.plot(x_vals, f_func(x_vals), label=r"$f(x) = e^x$", color="black", linewidth=2)  # Plot the original function

for n in n_values:
    # Compute Taylor polynomial and remainder
    taylor, remainder = taylor_polynomial(f, a, n)
    taylor_func = sp.lambdify(x, taylor, 'numpy')  # Convert Taylor polynomial to NumPy function

    # Print the results
    print(f"Taylor Polynomial P_{n}(x): {taylor}")
    print(f"Remainder R_{n}(x): {remainder}\n")

    # Plot the Taylor polynomial
    plt.plot(x_vals, taylor_func(x_vals), label=f"$P_{n}(x)$ (Degree {n})")

# Customize the plot
plt.title("Taylor's Theorem with Lagrange's Remainder for $e^x$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$ and $P_n(x)$")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.legend()
plt.grid()

# Show the plot
plt.show()