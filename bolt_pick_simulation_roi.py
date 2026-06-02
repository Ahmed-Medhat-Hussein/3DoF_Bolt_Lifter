import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from Kinematics.kinematics import Kinematics3DOF
import Kinematics.RobotConfig as cfg

# Import modules directly from your vision math and motion planners
from tools.Contour_detect import load_calibration, extract_screw_polygon, analyze_geometry, get_standard_screw_size
from motion.MotionPlan import MotionPlanner3DoF

# ── 1. CONFIGURATION & BATCH DROP MATRIX ─────────────────────────────────────
IMAGE_PATH          = 'Test_Images/12.jpeg'  
YOLO_WEIGHTS        = 'yolo_weights/best.pt'
CALIB_FILE_PATH     = "Camera_Calibration/camera_calibration.npz" # Calibration file integrated
MARKER_LENGTH       = 20.6              
MARKER_IN_BASE      = np.array([245, -45, 113.6])
BASE_CALIBRATION_ID = 0
TRANSFORM_MATRIX    = np.array([[0, 1], [1.0, 0]])
SCREW_Z_OFFSET      = -5
ROI_X, ROI_Y, ROI_W, ROI_H = 627, 140, 277, 208  # ROI Boundaries Preserved

# Pre-assigned destination drop coordinates matrix: (Base, Shoulder, Elbow)
SORTING_BIN_MAP = {
    "M6":  (-55, 10.0, -100.0),
    "M8":  (-45.0, 10, -80.0),
    "M10": (-35, 5.0, -30),
    "M12": (45, 20.0, -90.0),
    "M??": (45, 20.0, -90.0)  
}

# ── 2. COORDINATE TRANSFORM ENGINE ───────────────────────────────────────────
def calculate_pick_profile(u_anchor, v_anchor, u_screw, v_screw, pixel_scale, kinematics):
    delta_u   = u_screw - u_anchor
    delta_v   = v_screw - v_anchor
    dx_cam_mm = delta_u * pixel_scale
    dy_cam_mm = delta_v * pixel_scale
    robot_diff = np.dot(TRANSFORM_MATRIX, np.array([dx_cam_mm, dy_cam_mm]))
    X = MARKER_IN_BASE[0] + robot_diff[0]
    Y = MARKER_IN_BASE[1] + robot_diff[1]
    Z = MARKER_IN_BASE[2] + SCREW_Z_OFFSET
    angles = kinematics.inverse_kinematics(X, Y, Z)
    return angles, (X, Y, Z)

# ── 3. VIRTUAL TRACKING INTERPOLATOR ────────────────────────────────────────
class VisualPathCompiler:
    def __init__(self):
        self.current_joints = [0.0, 60.0, -135.0] # Mechanical Home pose initialization
        self.step_history = []

    def compile_simulated_sequence(self, sequences):
        for command, value in sequences:
            if command in ('relay', 'delay'): continue 
            
            target_pose = list(self.current_joints)
            if command == 'base': target_pose[0] = value
            elif command == 'shoulder': target_pose[1] = value
            elif command == 'elbow': target_pose[2] = value
            
            start = np.array(self.current_joints)
            end = np.array(target_pose)
            for i in range(1, 9): # 8 steps per transition interpolation frame
                interp = start + (end - start) * (i / 8)
                self.step_history.append(interp.tolist())
            self.current_joints = target_pose

# ── 4. ANIMATION DISPLAY INTERFACE ─────────────────────────────────────────
def draw_3d_robot(ax, angles, kinematics, status_msg):
    ax.clear()
    t1, t2, t3 = angles
    positions = kinematics.forward_kinematics(t1, t2, t3)
    px, py, pz = positions[:, 0], positions[:, 1], positions[:, 2]
    
    ax.plot(px, py, pz, 'o-', linewidth=4, markersize=8, color='blue', label='Simulated Arm')
    ax.set_title(f"{status_msg}\nBase:{t1:.1f}° | Shoulder:{t2:.1f}° | Elbow:{t3:.1f}°", color='purple', fontsize=9)
    
    limit = 450
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, limit)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper right')

# ── 5. RUN EXECUTOR ─────────────────────────────────────────────────────────
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not load simulation image target path: {IMAGE_PATH}")
        return

    # Load Calibration Matrix configuration elements
    mtx, dist = load_calibration(CALIB_FILE_PATH)
    if mtx is not None and dist is not None:
        print("[SIMULATION] Undistorting input file space geometry successfully.")
        img = cv2.undistort(img, mtx, dist, None, mtx)
    else:
        print("[SIMULATION] Warning: Proceeding without camera matrix correction data.")

    display_img = img.copy()
    kinematics = Kinematics3DOF()
    model = YOLO(YOLO_WEIGHTS)

    # Render static verification boundaries on simulation frame window
    cv2.rectangle(display_img, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 255), 2)

    # Detect ArUco Marker Anchor Planes
    if hasattr(cv2.aruco, 'ArucoDetector'):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(img)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(img, cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50))

    if ids is not None and BASE_CALIBRATION_ID in ids.flatten():
        idx = np.where(ids.flatten() == BASE_CALIBRATION_ID)[0][0]
        mc = corners[idx][0]
        u_anchor, v_anchor = float(np.mean(mc[:, 0])), float(np.mean(mc[:, 1]))
        pixel_scale = MARKER_LENGTH / np.linalg.norm(mc[0] - mc[1])
    else:
        print("[ERROR] Base context marker missing. Calibration aborted.")
        return

    # Slicing frame target area precisely to matching production ROI criteria
    roi_frame = img[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
    results = model(roi_frame, conf=0.4, verbose=False)
    compiled_tasks = []

    print("\n--- Starting ROI Vision Profiling Loop ---")
    for result in results:
        for box in result.boxes:
            # Map native crops back to standard frame space coordinates
            rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = rx1 + ROI_X, ry1 + ROI_Y, rx2 + ROI_X, ry2 + ROI_Y
            u_screw, v_screw = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            angles, _ = calculate_pick_profile(
                u_anchor, v_anchor, u_screw, v_screw, pixel_scale, kinematics
            )

            if angles is not None:
                # Local crop for Contour verification checking
                pad = 10
                cropped_img = img[max(0, int(y1)-pad):min(img.shape[0], int(y2)+pad), 
                                  max(0, int(x1)-pad):min(img.shape[1], int(x2)+pad)]
                polygon = extract_screw_polygon(cropped_img)
                size_str = "M??"
                
                if polygon is not None:
                    geom = analyze_geometry(polygon, cropped_img.shape)
                    size_num = get_standard_screw_size(geom[0]*pixel_scale, geom[1]*pixel_scale, geom[2]*pixel_scale)
                    size_str = f"M{size_num}"
                
                # Retrieve pre-assigned target dropdown tracking zones from our dictionary map
                drop_bin_pose = SORTING_BIN_MAP.get(size_str, SORTING_BIN_MAP["M??"])
                
                print(f"[TARGET REACHABLE] Type: {size_str} -> Pick Joint Params: {[round(a,1) for a in angles]} | Bin Drop: {drop_bin_pose}")
                compiled_tasks.append((angles, drop_bin_pose))
                
                # Visual graphics display feedback adjustments
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 120), 2)
                cv2.putText(display_img, size_str, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    # ── 6. PROCESS SEQUENCES THROUGH THE QUEUE PLANNER ───────────────────────
    planner = MotionPlanner3DoF(port='MOCK')
    compiler = VisualPathCompiler()

    nested_master_sequence = planner.run_multi_screw_sorting_cycle(compiled_tasks)
    compiler.compile_simulated_sequence(nested_master_sequence)

    # ── 7. PLAYBACK STREAM SELECTION ─────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Calibrated ROI Workspace Frame ({len(compiled_tasks)} screws)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122, projection='3d')
    print(f"\n[SIMULATION] Rendering compiled trajectory operations ({len(compiler.step_history)} updates)...")
    
    for step_joints in compiler.step_history:
        draw_3d_robot(ax2, step_joints, kinematics, "NESTED BATCH CYCLE IN TRANSIT...")
        plt.pause(0.01)

    print("[SUCCESS] Bounded, calibrated scene loop verified cleanly.")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()