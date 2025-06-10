import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd

import matplotlib.pyplot as plt

# Define the polar function r = f(theta)
def r_func(theta):
    # Example: r = 2 + np.sin(3*theta)
    return 2 + np.sin(3 * theta)

# Derivative dr/dtheta
def dr_dtheta(theta):
    return 3 * np.cos(3 * theta)

# Number of frames in the animation
n_frames = 100
theta_vals = np.linspace(0, 2 * np.pi, n_frames)
r_vals = r_func(theta_vals)
dr_vals = dr_dtheta(theta_vals)

# Calculate (x, y) for each theta
x_vals = r_vals * np.cos(theta_vals)
y_vals = r_vals * np.sin(theta_vals)

# Calculate tangent angle psi and angle phi between tangent and radius
psi_vals = theta_vals + np.arctan2(r_vals, dr_vals)
phi_vals = np.arctan2(r_vals, dr_vals)

# Prepare table of intermediate values
table = pd.DataFrame({
    'theta': theta_vals,
    'r': r_vals,
    'dr/dtheta': dr_vals,
    'x': x_vals,
    'y': y_vals,
    'psi (tangent angle)': psi_vals,
    'phi (angle tangent-radius)': phi_vals
})

print(table.head(10))  # Print first 10 rows as sample

# Animation
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, np.max(r_vals) + 1)
op_text = ax.text(0, 0, '', transform=ax.transAxes, fontsize=10, color='green', va='top', ha='left')
ot_text = ax.text(0, 0.05, '', transform=ax.transAxes, fontsize=10, color='blue', va='top', ha='left')
theta_text = ax.text(0, 0.10, '', transform=ax.transAxes, fontsize=10, color='red', va='top', ha='left')
phi_text = ax.text(0, 0.15, '', transform=ax.transAxes, fontsize=10, color='purple', va='top', ha='left')
psi_text = ax.text(0, 0.20, '', transform=ax.transAxes, fontsize=10, color='orange', va='top', ha='left')
point, = ax.plot([], [], 'ro', label='P(r, θ)')
tangent, = ax.plot([], [], 'b-', lw=2, label='Tangent PT')
radius, = ax.plot([], [], 'g--', lw=1, label='Radius OP')
ax.legend(loc='upper right')

def init():
    point.set_data([], [])
    tangent.set_data([], [])
    radius.set_data([], [])
    return point, tangent, radius

def animate(i):
    theta = theta_vals[i]
    r = r_vals[i]
    dr = dr_vals[i]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    psi = psi_vals[i]
    phi = phi_vals[i]

    # Point P
    point.set_data([theta], [r])

    # Tangent direction: (dr/dtheta, r)
    # Tangent vector in polar coordinates: (dr/dtheta, r)
    # To plot tangent, get two points along the tangent line at P
    # Slope in Cartesian: dy/dx = (dr/dtheta * sinθ + r * cosθ) / (dr/dtheta * cosθ - r * sinθ)
    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    # Two points along the tangent (in Cartesian)
    scale = 1.0
    x1 = x + scale * dx
    y1 = y + scale * dy
    x2 = x - scale * dx
    y2 = y - scale * dy

    # Convert back to polar for plotting
    theta1 = np.arctan2(y1, x1)
    r1 = np.hypot(x1, y1)
    theta2 = np.arctan2(y2, x2)
    r2 = np.hypot(x2, y2)

    tangent.set_data([theta1, theta2], [r1, r2])
    radius.set_data([0, theta], [0, r])

    return point, tangent, radius

ani = FuncAnimation(fig, animate, frames=n_frames, init_func=init, blit=True, interval=80, repeat=True)
plt.show()
# Add annotation handles

# Move annotation creation before plt.show() and update blit return values
# (Already done above, just ensure these are before FuncAnimation and plt.show())
# The annotation texts are updated in the animate function below.
def animate(i):
    theta = theta_vals[i]
    r = r_vals[i]
    dr = dr_vals[i]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    psi = psi_vals[i]
    phi = phi_vals[i]

    # Point P
    point.set_data([theta], [r])

    # Tangent direction
    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    scale = 1.0
    x1 = x + scale * dx
    y1 = y + scale * dy
    x2 = x - scale * dx
    y2 = y - scale * dy

    theta1 = np.arctan2(y1, x1)
    r1 = np.hypot(x1, y1)
    theta2 = np.arctan2(y2, x2)
    r2 = np.hypot(x2, y2)

    tangent.set_data([theta1, theta2], [r1, r2])
    radius.set_data([0, theta], [0, r])

    # Update annotation texts
    op_text.set_text(f"OP = {r:.2f}")
    ot_text.set_text(f"OT (tangent) = ψ = {psi:.2f} rad")
    theta_text.set_text(f"θ = {theta:.2f} rad")
    phi_text.set_text(f"φ = {phi:.2f} rad")
    psi_text.set_text(f"ψ = θ + arctan(r/dr) = {psi:.2f} rad")

    return point, tangent, radius, op_text, ot_text, theta_text, phi_text, psi_text

# Re-create animation with updated animate function and extra artists
ani = FuncAnimation(
    fig, animate, frames=n_frames, init_func=init,
    blit=True, interval=80, repeat=True
)
plt.show()