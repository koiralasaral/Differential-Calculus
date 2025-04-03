import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variables
x = sp.Symbol('x')

# Define the curve (example: y = x**2)
y = x**2

# Compute dy/dx
dy_dx = sp.diff(y, x)

# Compute the arc length integrand
arc_length_integrand = sp.sqrt(1 + dy_dx**2)

# Define the limits of integration
x_start = 0  # Start of the interval
x_end = 2    # End of the interval

# Compute the arc length
arc_length = sp.integrate(arc_length_integrand, (x, x_start, x_end))

# Display the result
print("Arc length:", arc_length)

# Plot the curve y = x^2
x_vals = np.linspace(x_start, x_end, 100)  # Generate x values
y_vals = x_vals**2  # Compute y values

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='$y = x^2$', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve of y = x^2')
plt.legend()
plt.grid()
plt.show()