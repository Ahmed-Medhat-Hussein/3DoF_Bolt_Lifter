import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Kinematics.kinematics import Kinematics3DOF
import Kinematics.RobotConfig as cfg

# Initialize Kinematics
kin = Kinematics3DOF()

# Create the figure and 3D axis
fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.25) # Leave space for sliders
ax = fig.add_subplot(111, projection='3d')

# Initial target position
init_x, init_y, init_z = 150, 100, 300

def draw_robot(x, y, z):
    ax.clear()
    
    # Plot target point
    ax.scatter([x], [y], [z], color='red', s=100, label='Target', zorder=5)
    
    # Calculate Inverse Kinematics
    sol = kin.inverse_kinematics(x, y, z)
    
    if sol:
        t1, t2, t3 = sol
        # Get joint positions via Forward Kinematics
        positions = kin.forward_kinematics(t1, t2, t3)
        px, py, pz = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Draw arm links
        ax.plot(px, py, pz, 'o-', linewidth=4, markersize=8, color='blue', label='Robot Arm')
        
        # Annotate joint angles
        status = f"REACHABLE\nt1={t1:.1f}°, t2={t2:.1f}°, t3={t3:.1f}°"
        ax.set_title(status, color='green', fontsize=12)
    else:
        # Show only a ghost of the base-to-shoulder if unreachable
        ax.plot([0, 0], [0, 0], [0, cfg.L1], 'o--', color='gray', alpha=0.3)
        ax.set_title("UNREACHABLE / LIMIT VIOLATION", color='red', fontsize=12)

    # Set plot boundaries and labels
    limit = 450
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, limit)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper right')

# --- Define Sliders ---
ax_x = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_y = plt.axes([0.2, 0.10, 0.65, 0.03])
ax_z = plt.axes([0.2, 0.05, 0.65, 0.03])

samp_x = Slider(ax_x, 'X', -400.0, 400.0, valinit=init_x)
samp_y = Slider(ax_y, 'Y', -400.0, 400.0, valinit=init_y)
samp_z = Slider(ax_z, 'Z', 0.0, 600.0, valinit=init_z)

def update(val):
    draw_robot(samp_x.val, samp_y.val, samp_z.val)
    fig.canvas.draw_idle()

# Register the update function with sliders
samp_x.on_changed(update)
samp_y.on_changed(update)
samp_z.on_changed(update)

# Initial draw
draw_robot(init_x, init_y, init_z)

plt.show()