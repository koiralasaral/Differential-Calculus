import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

# Define the variable and the curve
x_sym = sp.Symbol('x')
y_sym = x_sym**3 

# Compute dy/dx (symbolic)
dy_dx_sym = sp.diff(y_sym, x_sym)

# Convert SymPy expressions to NumPy functions
f = sp.lambdify(x_sym, y_sym, 'numpy')
f_prime = sp.lambdify(x_sym, dy_dx_sym, 'numpy')

# Define tangent and normal equations
def tangent_eq(x, x_point, y_point, dy_dx_point):
    return dy_dx_point * (x - x_point) + y_point

def normal_eq(x, x_point, y_point, dy_dx_point):
    slope_normal = -1 / dy_dx_point  # Normal is perpendicular to tangent
    return slope_normal * (x - x_point) + y_point

# Choose a specific point (x, y)
x_point = 2  # Change this value for a different point

# Calculate the y-value at the specified x-point
y_point = f(x_point)# On the curve y = x^2

# Evaluate dy/dx at the given point
dy_dx_point = sp.lambdify(x_sym, dy_dx_sym, 'numpy')(x_point)

# Generate values for tangent and normal
x_vals_tangent_normal = np.linspace(0, 3, 100)  # Range for tangent/normal lines
tangent_vals = tangent_eq(x_vals_tangent_normal, x_point, y_point, dy_dx_point)
normal_vals = normal_eq(x_vals_tangent_normal, x_point, y_point, dy_dx_point)

# Calculate geometric properties at the given point
subtangent_val = y_point / dy_dx_point
subnormal_val = y_point * dy_dx_point
tangent_length_val = y_point * np.sqrt(1 + (1/dy_dx_point**2))
normal_length_val = y_point * np.sqrt(1 + dy_dx_point**2)

# Print the calculated values
print(f"At point ({x_point}, {y_point}):")
print(f"Subtangent: {subtangent_val}")
print(f"Subnormal: {subnormal_val}")
print(f"Length of Tangent: {tangent_length_val}")
print(f"Length of Normal: {normal_length_val}")

# Plot the curve and geometric propertie
x_vals_curve = np.linspace(-5, 5, 1000)
y_vals_curve = f(x_vals_curve)

plt.figure(figsize=(10, 10))

# Plot the curve
plt.plot(x_vals_curve, y_vals_curve, label=f'Curve: y = x^{3}', color='black', linestyle='--')

# Plot the specific point
plt.scatter(x_point, y_point, color='red', label=f'Point: ({x_point}, {y_point})')

# Plot the tangent
plt.plot(x_vals_tangent_normal, tangent_vals, label='Tangent', color='orange', linestyle='-')

# Plot the normal
plt.plot(x_vals_tangent_normal, normal_vals, label='Normal', color='purple', linestyle='-')

# Annotate subtangent and subnormal
plt.arrow(x_point, y_point, subtangent_val, 0, color='blue', head_width=0.1, label='Subtangent')
plt.arrow(x_point, y_point, 0, -subnormal_val, color='green', head_width=0.1, label='Subnormal')

# Display the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Geometric Properties of the Curve y = x^3 at a Specific Point')
plt.legend()
plt.grid()
# Prepare intermediate values for the table
data = {
    'Property': [
        'x_point',
        'y_point',
        'dy_dx_point',
        'Subtangent',
        'Subnormal',
        'Length of Tangent',
        'Length of Normal'
    ],
    'Value': [
        x_point,
        y_point,
        dy_dx_point,
        subtangent_val,
        subnormal_val,
        tangent_length_val,
        normal_length_val
    ]
}

df = pd.DataFrame(data)
print("\nCalculated Intermediate Values:")
print(df.to_string(index=False))

plt.show()
# Calculate geometric properties for various angles (theta)
# For y = x^3, the angle theta is the angle of the tangent with the x-axis: tan(theta) = dy/dx

angles_deg = np.arange(0, 7200, 10)  # 0 to 7200 degrees in steps of 10
results = {
    'Angle (deg)': [],
    'x_point': [],
    'y_point': [],
    'dy/dx': [],
    'Subtangent': [],
    'Subnormal': [],
    'Length of Tangent': [],
    'Length of Normal': []
}

for angle in angles_deg:
    theta_rad = np.deg2rad(angle)
    # dy/dx = tan(theta)
    if angle == 90:
        continue  # dy/dx is infinite, skip vertical tangent
    dy_dx_val = np.tan(theta_rad)
    # For y = x^3, dy/dx = 3x^2 => x = sqrt(dy/dx / 3)
    if dy_dx_val < 0:
        continue  # x would be imaginary for negative slopes
    x_val = np.sqrt(dy_dx_val / 3) if dy_dx_val != 0 else 0
    y_val = x_val ** 3

    # Subtangent = y / (dy/dx)
    subtangent = y_val / dy_dx_val if dy_dx_val != 0 else np.nan
    # Subnormal = y * dy/dx
    subnormal = y_val * dy_dx_val
    # Length of tangent = y * sqrt(1 + (1/(dy/dx)^2))
    tangent_length = y_val * np.sqrt(1 + (1 / dy_dx_val ** 2)) if dy_dx_val != 0 else np.nan
    # Length of normal = y * sqrt(1 + (dy/dx)^2)
    normal_length = y_val * np.sqrt(1 + dy_dx_val ** 2)

    results['Angle (deg)'].append(angle)
    results['x_point'].append(x_val)
    results['y_point'].append(y_val)
    results['dy/dx'].append(dy_dx_val)
    results['Subtangent'].append(subtangent)
    results['Subnormal'].append(subnormal)
    results['Length of Tangent'].append(tangent_length)
    results['Length of Normal'].append(normal_length)

# Show the table
df_angles = pd.DataFrame(results)
print("\nGeometric Properties at Various Tangent Angles:")
print(df_angles.to_string(index=False))
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 10))

def animate(i):
    ax.clear()
    # Plot the curve
    ax.plot(x_vals_curve, y_vals_curve, label='Curve: y = x^3', color='black', linestyle='--')
    # Get current values
    angle = df_angles['Angle (deg)'][i]
    x_val = df_angles['x_point'][i]
    y_val = df_angles['y_point'][i]
    dy_dx_val = df_angles['dy/dx'][i]
    subtangent = df_angles['Subtangent'][i]
    subnormal = df_angles['Subnormal'][i]
    tangent_length = df_angles['Length of Tangent'][i]
    normal_length = df_angles['Length of Normal'][i]

    # Plot the specific point
    ax.scatter(x_val, y_val, color='red', label=f'Point: ({x_val:.2f}, {y_val:.2f})')

    # Tangent and normal lines
    if dy_dx_val != 0:
        tangent_x = np.linspace(x_val - 1, x_val + 1, 100)
        tangent_y = dy_dx_val * (tangent_x - x_val) + y_val
        ax.plot(tangent_x, tangent_y, color='orange', label='Tangent')
        normal_slope = -1 / dy_dx_val
        normal_x = np.linspace(x_val - 1, x_val + 1, 100)
        normal_y = normal_slope * (normal_x - x_val) + y_val
        ax.plot(normal_x, normal_y, color='purple', label='Normal')
    else:
        # Vertical tangent
        ax.axvline(x_val, color='orange', label='Tangent')
        ax.axhline(y_val, color='purple', label='Normal')

    # Subtangent (horizontal arrow)
    if not np.isnan(subtangent):
        ax.arrow(x_val, y_val, subtangent, 0, color='blue', head_width=0.1, length_includes_head=True)
        ax.text(x_val + subtangent/2, y_val + 0.5, 'Subtangent', color='blue')

    # Subnormal (vertical arrow)
    ax.arrow(x_val, y_val, 0, -subnormal, color='green', head_width=0.1, length_includes_head=True)
    ax.text(x_val + 0.1, y_val - subnormal/2, 'Subnormal', color='green')

    # Annotate lengths
    ax.text(x_val, y_val + 1, f'Angle: {angle}°', fontsize=12, color='black')
    ax.text(x_val, y_val + 0.5, f'Tangent Length: {tangent_length:.2f}', color='orange')
    ax.text(x_val, y_val + 0.2, f'Normal Length: {normal_length:.2f}', color='purple')

    ax.set_xlim(-1, 3)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Geometric Properties at Various Tangent Angles')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, animate, frames=len(df_angles), interval=1000, repeat=True)
plt.show()