import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constant dx, and dy varying from -2 to 2 for demonstration
dx = 1.0  # constant horizontal step
dy_values = np.linspace(-2, 2, 100)  # 100 steps from -2 to 2

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)

def init():
    """Initialize the background of the animation."""
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    return []

def animate(i):
    """Update the animation for frame i."""
    # Get current dy and calculate phi, ds, sin and cos of phi.
    dy = dy_values[i]
    # Here, tan(phi) = dy/dx, so phi is:
    phi = np.arctan2(dy, dx)
    # ds = sqrt(dx^2 + dy^2)
    ds = np.sqrt(dx**2 + dy**2)
    # cos(phi) = dx/ds and sin(phi) = dy/ds
    cos_phi = dx / ds
    sin_phi = dy / ds

    # Clear previous drawings
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True)

    # Draw vectors:
    # 1. Draw dx as a blue arrow from (0,0) to (dx, 0)
    ax.arrow(0, 0, dx, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue', label='dx')
    # 2. Draw dy as a green arrow starting at (dx,0) going vertically to (dx, dy)
    ax.arrow(dx, 0, 0, dy, head_width=0.1, head_length=0.2, fc='green', ec='green', label='dy')
    # 3. Draw ds as a red arrow from (0,0) to (dx, dy)
    ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.2, fc='red', ec='red', label='ds')

    # Optionally, draw a unit circle for orientation reference.
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Display computed values on the plot:
    info = (f"dy = {dy:.2f}\n"
            f"φ = {np.degrees(phi):.2f}°\n"
            f"ds = {ds:.2f}\n"
            f"cos(φ) = {cos_phi:.2f}\n"
            f"sin(φ) = {sin_phi:.2f}")
    ax.text(-2.8, 2.5, info, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Also print the intermediate values to the console:
    print(f"Step {i:3d}: dx = {dx:.2f}, dy = {dy:.2f}, ds = {ds:.2f}, "
          f"φ = {phi:.2f} rad, cos(φ) = {cos_phi:.2f}, sin(φ) = {sin_phi:.2f}")

    return []

# Create the animation, with a short delay between frames (interval in milliseconds)
ani = animation.FuncAnimation(fig, animate, frames=len(dy_values),
                              init_func=init, interval=100, blit=False)

plt.show()