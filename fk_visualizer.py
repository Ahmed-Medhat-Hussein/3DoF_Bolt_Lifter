# Interactive_visualizer.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Kinematics.kinematics import Kinematics3DOF
import Kinematics.RobotConfig as cfg

def main():
    kin = Kinematics3DOF()
    
    init_t1 = 0.0
    init_t2 = 60.0
    init_t3 = -135.0
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3) 
    ax = fig.add_subplot(111, projection='3d')
    
    positions = kin.forward_kinematics(init_t1, init_t2, init_t3)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]
    
    # --- NEW: Plot the arm and camera separately ---
    # Plot standard arm joints (indices 0 to 3)
    arm_line, = ax.plot(x_coords[:4], y_coords[:4], z_coords[:4], '-o', color='blue', markersize=8, linewidth=4, label="Arm")
    
    
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_zlim([0, 400])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Interactive 3-DOF Robot Visualizer')
    ax.legend()

    ax_t1 = plt.axes([0.2, 0.2, 0.65, 0.03])
    ax_t2 = plt.axes([0.2, 0.15, 0.65, 0.03])
    ax_t3 = plt.axes([0.2, 0.1, 0.65, 0.03])
    
    lim1 = cfg.JOINT_LIMITS['theta1']
    lim2 = cfg.JOINT_LIMITS['theta2']
    lim3 = cfg.JOINT_LIMITS['theta3']

    slider_t1 = Slider(ax_t1, 'Base (T1)', lim1[0], lim1[1], valinit=init_t1)
    slider_t2 = Slider(ax_t2, 'Shoulder (T2)', lim2[0], lim2[1], valinit=init_t2)
    slider_t3 = Slider(ax_t3, 'Elbow (T3)', lim3[0], lim3[1], valinit=init_t3)

    def update(val):
        t1 = slider_t1.val
        t2 = slider_t2.val
        t3 = slider_t3.val
        
        new_positions = kin.forward_kinematics(t1, t2, t3)
        
        arm_line.set_data(new_positions[:4, 0], new_positions[:4, 1])
        arm_line.set_3d_properties(new_positions[:4, 2])
        
        fig.canvas.draw_idle()

    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    slider_t3.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()