import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc

# --------------------------------------------------
# Define the polar function and its derivative
# --------------------------------------------------
def r_func(theta):
    # Example polar function: r = exp(π*sin(θ))
    return np.exp(np.pi * np.sin(theta))

def dr_dtheta(theta):
    # Derivative: dr/dθ = π e^(π*sin(θ)) cos(θ)
    return np.pi * np.exp(np.pi * np.sin(theta)) * np.cos(theta)

# --------------------------------------------------
# Pre-compute data for animation
# --------------------------------------------------
n_frames = 100
theta_vals = np.linspace(0, 2 * np.pi, n_frames)
r_vals    = r_func(theta_vals)
dr_vals   = dr_dtheta(theta_vals)

# Cartesian coordinates for the static curve (and for computing dynamic values)
x_vals = r_vals * np.cos(theta_vals)
y_vals = r_vals * np.sin(theta_vals)

# Compute angles:
# Let φ be the angle between the radius and the tangent
phi_vals = np.arctan2(r_vals, dr_vals)   # φ = arctan(r/dr)
# The tangent (absolute) angle is then: ψ = θ + φ
psi_vals = theta_vals + phi_vals

# --------------------------------------------------
# Print table and array of [theta, φ, ψ] values
# --------------------------------------------------
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

# --------------------------------------------------
# Create figure and subplots:
#  - Left: Polar plot with dynamic annotations and arc for φ
#  - Right: Cartesian plot with static curve and dynamic elements.
# --------------------------------------------------
fig = plt.figure(figsize=(16, 8))
# Polar subplot (left)
ax_polar = fig.add_subplot(121, projection='polar')
ax_polar.set_title("Polar Animation")
ax_polar.set_xlim(0, 2 * np.pi)
ax_polar.set_ylim(0, np.max(r_vals) + 1)

# Cartesian subplot (right)
ax_cart = fig.add_subplot(122)
ax_cart.set_title("Cartesian Animation")
ax_cart.plot(x_vals, y_vals, 'k-', lw=1, label="Static Curve")  # Static full curve
ax_cart.set_xlabel("x")
ax_cart.set_ylabel("y")
ax_cart.grid(True)
ax_cart.set_aspect('equal', 'datalim')

# --------------------------------------------------
# Polar plot artists and annotations
# --------------------------------------------------
# Dynamic point on polar curve
point, = ax_polar.plot([], [], 'ro', label='P (r, θ)')
# Tangent line PT (computed in Cartesian, converted back to polar)
tangent, = ax_polar.plot([], [], 'b-', lw=2, label='Tangent PT')
# Radius OP from origin to point P
radius, = ax_polar.plot([], [], 'g--', lw=1, label='Radius OP')
ax_polar.legend(loc='upper right')

# Annotation texts in polar subplot
op_text    = ax_polar.text(0, 0, '', transform=ax_polar.transAxes,
                           fontsize=10, color='green', va='top', ha='left')
ot_text    = ax_polar.text(0, 0.05, '', transform=ax_polar.transAxes,
                           fontsize=10, color='blue', va='top', ha='left')
theta_text = ax_polar.text(0, 0.10, '', transform=ax_polar.transAxes,
                           fontsize=10, color='red', va='top', ha='left')
phi_text   = ax_polar.text(0, 0.15, '', transform=ax_polar.transAxes,
                           fontsize=10, color='purple', va='top', ha='left')
psi_text   = ax_polar.text(0, 0.20, '', transform=ax_polar.transAxes,
                           fontsize=10, color='orange', va='top', ha='left')

# Arc for visualizing φ angle. Drawn as a small arc at the origin.
arc_radius = 0.5  # Fixed radius for the arc demonstration.
arc_phi = Arc((0, 0), width=arc_radius, height=arc_radius,
              angle=0, theta1=0, theta2=0, color='magenta', lw=2)
ax_polar.add_patch(arc_phi)

# --------------------------------------------------
# Cartesian plot artists and annotations
# --------------------------------------------------
# Dynamic point on the Cartesian curve
point_cart, = ax_cart.plot([], [], 'ro', label="P")
# Tangent line in Cartesian form
tangent_cart, = ax_cart.plot([], [], 'b-', lw=2, label="Tangent")
# Radius line from (0,0) to the point in Cartesian form
radius_cart, = ax_cart.plot([], [], 'g--', lw=1, label="OP")
# Annotation text for Cartesian values
point_info_cart = ax_cart.text(0.05, 0.95, '', transform=ax_cart.transAxes,
                               fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
ax_cart.legend(loc='upper right')

# --------------------------------------------------
# Initialization function for animation (reset animated artists)
# --------------------------------------------------
def init():
    # Polar animated objects:
    point.set_data([], [])
    tangent.set_data([], [])
    radius.set_data([], [])
    op_text.set_text('')
    ot_text.set_text('')
    theta_text.set_text('')
    phi_text.set_text('')
    psi_text.set_text('')
    arc_phi.theta1 = 0
    arc_phi.theta2 = 0

    # Cartesian animated objects:
    point_cart.set_data([], [])
    tangent_cart.set_data([], [])
    radius_cart.set_data([], [])
    point_info_cart.set_text('')
    
    # Return all artists for blitting
    return (point, tangent, radius, op_text, ot_text, theta_text, phi_text, psi_text, arc_phi,
            point_cart, tangent_cart, radius_cart, point_info_cart)

# --------------------------------------------------
# Animation update function (called for each frame)
# --------------------------------------------------
def animate(i):
    # Get current polar values for frame i
    theta = theta_vals[i]
    r     = r_vals[i]
    dr    = dr_vals[i]
    psi   = psi_vals[i]
    phi   = phi_vals[i]

    # --- Update Polar Plot ---
    # Update point on polar curve (in polar coordinates: (θ, r))
    point.set_data([theta], [r])
    
    # Compute the Cartesian coordinates of the current point for tangent computation.
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Compute derivatives for the tangent vector in Cartesian coordinates:
    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    norm = np.hypot(dx, dy)
    if norm != 0:
        dx, dy = dx / norm, dy / norm  # Normalize the tangent direction

    # Define two points along the tangent (centered at (x,y))
    scale = 1.0  # Adjust the tangent line length as necessary
    x1 = x + scale * dx
    y1 = y + scale * dy
    x2 = x - scale * dx
    y2 = y - scale * dy

    # Sample points along the tangent in Cartesian coordinates and convert to polar for the polar subplot
    t_vals = np.linspace(-1, 1, 20)  # 20 points along the tangent
    x_tan = x + t_vals * dx
    y_tan = y + t_vals * dy
    theta_tan = np.arctan2(y_tan, x_tan)
    r_tan = np.hypot(x_tan, y_tan)
    tangent.set_data(theta_tan, r_tan)
    
    # Update the radius OP (draw from origin to the current point)
    radius.set_data([0, theta], [0, r])
    
    # Update the arc for φ: the arc starts at θ (in degrees) and spans φ (in degrees)
    phi_deg = abs(np.degrees(psi - theta))  # ψ - θ is φ; take absolute value for the span
    start_deg = np.degrees(theta)
    arc_phi.theta1 = start_deg
    arc_phi.theta2 = start_deg + phi_deg

    # Update annotation texts in the polar subplot
    op_text.set_text(f"OP = {r:.2f}")
    ot_text.set_text(f"OT (ψ) = {psi:.2f} rad")
    theta_text.set_text(f"θ = {theta:.2f} rad")
    phi_text.set_text(f"φ = {phi:.2f} rad")
    psi_text.set_text(f"ψ = θ + φ = {psi:.2f} rad")
    
    # --- Update Cartesian Plot ---
    # Compute the Cartesian coordinates for the dynamic point
    point_cart.set_data([x], [y])
    # For Cartesian tangent, use the same computed derivatives (dx, dy)
    # Define two endpoints along the tangent line in Cartesian coordinates:
    x1_cart = x + scale * dx
    y1_cart = y + scale * dy
    x2_cart = x - scale * dx
    y2_cart = y - scale * dy
    tangent_cart.set_data([x1_cart, x2_cart], [y1_cart, y2_cart])
    # Update the radius (line from origin to (x,y))
    radius_cart.set_data([0, x], [0, y])
    # Update the annotation text for Cartesian plot with current values.
    point_info_cart.set_text(f"θ = {theta:.2f} rad\nOP = r = {r:.2f}\nφ = {phi:.2f} rad\nψ = {psi:.2f} rad")
    
    return (point, tangent, radius, op_text, ot_text, theta_text, phi_text, psi_text, arc_phi,
            point_cart, tangent_cart, radius_cart, point_info_cart)

# --------------------------------------------------
# Create and run the animation
# --------------------------------------------------
ani = FuncAnimation(fig, animate, frames=n_frames,
                    init_func=init, blit=True, interval=80, repeat=True)

plt.tight_layout()
plt.show()