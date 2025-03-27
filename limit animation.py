import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function
def func(x):
    return np.sin(x) / x

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)  # Exclude zero to avoid division by zero
x = x[x != 0]  # Remove x=0 to prevent undefined values
y = func(x)
ax.plot(x, y, label=r'$\frac{\sin(x)}{x}$')
ax.axhline(1, color='red', linestyle='--', label='Limit as $x \to 0$')
ax.legend()
ax.set_xlim(-10, 10)
ax.set_ylim(-0.5, 1.5)
ax.set_title(r'Visualizing the Limit of $\frac{\sin(x)}{x}$ as $x \to 0$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Create a dot for animated visualization
point, = ax.plot([], [], 'bo', label='Approaching $x \to 0$')

# Initialize animation
def init():
    point.set_data([], [])
    return point,

# Update the animation frame
def update(frame):
    x_val = frame  # Current x value
    y_val = func(x_val) if x_val != 0 else 1  # Avoid division by zero
    point.set_data([x_val], [y_val])  # Wrap x_val and y_val in lists
    return point,

# Create animation
frames = np.linspace(-10, 10, 200)
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

plt.show()