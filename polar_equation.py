import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the polar function and its derivative
def r_func(theta):
    return 2 + 2 * np.sin(theta)

def dr_dtheta(theta):
    return 2 * np.cos(theta)

# Generate data for the polar curve and extra computations
theta_vals = np.linspace(0, 2 * np.pi, 1000)
r_vals = r_func(theta_vals)
phi_vals = np.arctan2(r_func(theta_vals) * dr_dtheta(theta_vals), r_func(theta_vals))
psi_vals = phi_vals + theta_vals

# Create a DataFrame of the computed values (for reference, though we'll show only the slider-selected row interactively)
table_data = pd.DataFrame({
    'theta': theta_vals,
    'r': r_vals,
    'phi': phi_vals,
    'psi': psi_vals
})
print("Sample Table Data (first 10 rows):")
print(table_data.head(10))

# Set up the figure with two panels:
# Left: Polar plot with dynamic point and tangent indication,
# Right: Table showing current computed values.
fig = plt.figure(figsize=(12, 6))

# Left subplot: Polar plot
ax_polar = fig.add_subplot(121, polar=True)
ax_polar.set_title("Polar Curve with Dynamic Point")
# Plot the static polar curve
ax_polar.plot(theta_vals, r_vals, label="r = 2 + 2*sin(θ)", color='blue')
# Add a dynamic point (initially at θ=0)
(point_line,) = ax_polar.plot([], [], 'ro', label="Current Point")

# Right subplot: Table display (we will update a table dynamically)
ax_table = fig.add_subplot(122)
ax_table.axis('off')
ax_table.set_title("Computed Values", fontweight='bold')
# Initially, display the header and first row (θ=0)
cell_text = [['θ', 'r', 'φ', 'ψ'], 
             [f"{theta_vals[0]:.2f}", f"{r_vals[0]:.2f}", f"{phi_vals[0]:.2f}", f"{psi_vals[0]:.2f}"]]
table_plot = ax_table.table(cellText=cell_text, loc='center', cellLoc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(12)
table_plot.scale(1, 2)

# Add an interactive slider below the figure for selecting θ interactively.
slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.04])
theta_slider = Slider(ax=slider_ax, label='θ', valmin=0, valmax=2 * np.pi, valinit=0, valfmt='%.2f')

def update_visualization(val):
    # Get the current theta value from the slider.
    curr_theta = theta_slider.val
    curr_r = r_func(curr_theta)
    
    # --- Update the polar plot ---
    # Set the dynamic point on the polar curve.
    point_line.set_data([curr_theta], [curr_r])
    
    # Approximate the tangent line (in Cartesian coordinates) at the chosen angle.
    # Convert polar (r, θ) to Cartesian (x, y)
    x0 = curr_r * np.cos(curr_theta)
    y0 = curr_r * np.sin(curr_theta)
    # Derivatives: dx/dθ and dy/dθ (using chain rule)
    dr = dr_dtheta(curr_theta)
    dx_dtheta = dr * np.cos(curr_theta) - curr_r * np.sin(curr_theta)
    dy_dtheta = dr * np.sin(curr_theta) + curr_r * np.cos(curr_theta)
    # Normalize for display (scale factor)
    scale = 0.5
    dx = dx_dtheta
    dy = dy_dtheta
    norm = np.hypot(dx, dy)
    if norm != 0:
        dx, dy = dx / norm * scale, dy / norm * scale
    # Remove any previous tangent annotations by clearing and replotting the static curve and point.
    ax_polar.lines = ax_polar.lines[:2]  # Keep the original curve and dynamic point.
    # Add a new arrow annotation showing the tangent direction (displayed in Cartesian coordinates).
    # For polar axes, we convert the start and end polar coordinates.
    # Starting point (x0,y0) and ending point (x0+dx, y0+dy): 
    # convert them back into polar coordinates.
    r_start = np.hypot(x0, y0)
    theta_start = np.arctan2(y0, x0)
    x_end = x0 + dx
    y_end = y0 + dy
    r_end = np.hypot(x_end, y_end)
    theta_end = np.arctan2(y_end, x_end)
    ax_polar.annotate('', xy=(theta_end, r_end), xytext=(theta_start, r_start),
                      arrowprops=dict(facecolor='green', shrink=0.05))

    # --- Update the table on the right ---
    # Prepare updated row for the current theta.
    curr_phi = np.arctan2(curr_r * dr, curr_r)  # More illustrative comparison (may be refined)
    curr_psi = curr_phi + curr_theta
    new_cell_text = [['θ', 'r', 'φ', 'ψ'],
                     [f"{curr_theta:.2f}", f"{curr_r:.2f}", f"{curr_phi:.2f}", f"{curr_psi:.2f}"]]
    ax_table.clear()  # Clear the previous table
    ax_table.axis('off')
    # Redraw the table with new values.
    ax_table.table(cellText=new_cell_text, loc='center', cellLoc='center')
    ax_table.set_title("Computed Values", fontweight='bold')
    
    fig.canvas.draw_idle()

# Hook up the slider with the update function.
theta_slider.on_changed(update_visualization)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()