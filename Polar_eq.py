import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc

# -------------------------------
# Define the function and derivative
# -------------------------------
def r_func(theta):
    # Example polar function: r = 2 + sin(3θ)
    return 2 + np.sin(3 * theta)

def dr_dtheta(theta):
    # Derivative: dr/dθ = 3 cos(3θ)
    return 3 * np.cos(3 * theta)

# -------------------------------
# Animation setup and data computation
# -------------------------------
n_frames = 100
theta_vals = np.linspace(0, 2 * np.pi, n_frames)
r_vals    = r_func(theta_vals)
dr_vals   = dr_dtheta(theta_vals)

# Compute Cartesian coordinates (for reference, if needed)
x_vals = r_vals * np.cos(theta_vals)
y_vals = r_vals * np.sin(theta_vals)

# Compute tangent-angle information
# Let φ be defined so that tan(φ) = r/(dr/dθ). Using arctan2 ensures proper quadrant.
phi_vals = np.arctan2(r_vals, dr_vals)
# The tangent line direction, ψ, is then given by:
psi_vals = theta_vals + phi_vals

# Print sample table and arrays of values
table = pd.DataFrame({
    'theta': theta_vals,
    'r': r_vals,
    'dr/dtheta': dr_vals,
    'x': x_vals,
    'y': y_vals,
    'phi (angle between radius and tangent)': phi_vals,
    'psi (tangent angle)': psi_vals
})
print("Sample Table (first 10 rows):")
print(table.head(10))
print("\nArray of [theta, phi, psi] values:")
print(np.column_stack((theta_vals, phi_vals, psi_vals)))

# -------------------------------
# Set up the figure with a polar subplot
# -------------------------------
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, np.max(r_vals) + 1)

# -------------------------------
# Annotations for dynamic values
# -------------------------------
op_text    = ax.text(0, 0, '', transform=ax.transAxes,
                     fontsize=10, color='green', va='top', ha='left')
ot_text    = ax.text(0, 0.05, '', transform=ax.transAxes,
                     fontsize=10, color='blue', va='top', ha='left')
theta_text = ax.text(0, 0.10, '', transform=ax.transAxes,
                     fontsize=10, color='red', va='top', ha='left')
phi_text   = ax.text(0, 0.15, '', transform=ax.transAxes,
                     fontsize=10, color='purple', va='top', ha='left')
psi_text   = ax.text(0, 0.20, '', transform=ax.transAxes,
                     fontsize=10, color='orange', va='top', ha='left')

# -------------------------------
# Plot elements: the point, tangent and radius lines.
# -------------------------------
# Point P on the curve
point, = ax.plot([], [], 'ro', label='P (r, θ)')
# Tangent line PT
tangent, = ax.plot([], [], 'b-', lw=2, label='Tangent PT')
# Radius line OP from origin to P
radius, = ax.plot([], [], 'g--', lw=1, label='Radius OP')
ax.legend(loc='upper right')

# -------------------------------
# Create an Arc patch to illustrate the φ angle.
# We'll show it as a small arc (with fixed radius for visibility) centered at the origin.
# The arc will start at angle θ (in degrees) and span φ (in degrees).
# (Note: This is a visual aid; φ is the angle between the radius and tangent.)
arc_radius = 0.5  # Set a fixed small radius for the arc
arc_phi = Arc((0, 0), width=arc_radius, height=arc_radius,
              angle=0, theta1=0, theta2=0, color='magenta', lw=2)
ax.add_patch(arc_phi)

# -------------------------------
# Initialization function for the animation
# -------------------------------
def init():
    point.set_data([], [])
    tangent.set_data([], [])
    radius.set_data([], [])
    op_text.set_text('')
    ot_text.set_text('')
    theta_text.set_text('')
    phi_text.set_text('')
    psi_text.set_text('')
    # Initialize arc
    arc_phi.theta1 = 0
    arc_phi.theta2 = 0
    return point, tangent, radius, op_text, ot_text, theta_text, phi_text, psi_text, arc_phi

# -------------------------------
# Animation function that updates each frame.
# -------------------------------
def animate(i):
    # Current values for frame i
    theta = theta_vals[i]
    r     = r_vals[i]
    dr    = dr_vals[i]
    psi   = psi_vals[i]
    phi   = phi_vals[i]  # φ computed as arctan2(r, dr)

    # Update the point on the polar plot: (θ, r)
    point.set_data([theta], [r])
    
    # --- Update tangential computations ---
    # In Cartesian, the point is (x = r cosθ, y = r sinθ)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Derivatives to compute the tangent direction:
    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    norm = np.hypot(dx, dy)
    if norm != 0:
        dx, dy = dx / norm, dy / norm  # Unit direction vector

    # Define two points along the tangent (centered at the current point)
    scale = 1.0  # You may adjust this for visual clarity
    x1 = x + scale * dx
    y1 = y + scale * dy
    x2 = x - scale * dx
    y2 = y - scale * dy
    # Convert these two Cartesian points back to polar coordinates:
    theta1 = np.arctan2(y1, x1)
    r1     = np.hypot(x1, y1)
    theta2 = np.arctan2(y2, x2)
    r2     = np.hypot(x2, y2)
    tangent.set_data([theta1, theta2], [r1, r2])
    
    # Update the radius OP
    radius.set_data([0, theta], [0, r])
    
    # --- Update the arc representing φ ---
    # We want to display the angle between the radius (OP) and the tangent.
    # Here, we use the difference between the tangent direction and the radial direction.
    # Since the tangent direction is ψ = θ + φ, we can set:
    phi_deg = np.degrees(np.abs(psi - theta))
    # Set the arc to start at theta (in degrees) and extend φ_deg
    start_deg = np.degrees(theta)
    arc_phi.center = (0, 0)      # arc centered at origin
    arc_phi.width = arc_radius  # fixed small arc for demonstration
    arc_phi.height = arc_radius
    arc_phi.theta1 = start_deg
    arc_phi.theta2 = start_deg + phi_deg

    # --- Update annotation texts ---
    op_text.set_text(f"OP = {r:.2f}")
    ot_text.set_text(f"OT (Tangent, ψ) = {psi:.2f} rad")
    theta_text.set_text(f"θ = {theta:.2f} rad")
    phi_text.set_text(f"φ = {phi:.2f} rad")
    psi_text.set_text(f"ψ = θ + φ = {psi:.2f} rad")

    return point, tangent, radius, op_text, ot_text, theta_text, phi_text, psi_text, arc_phi

# -------------------------------
# Create and start the animation
# -------------------------------
ani = FuncAnimation(
    fig, animate, frames=n_frames,
    init_func=init, blit=True, interval=80, repeat=True
)

plt.show()