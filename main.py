import cv2
import numpy as np
import time
from ultralytics import YOLO
from Kinematics.kinematics import Kinematics3DOF
import Kinematics.RobotConfig as cfg
from motion.MotionPlan import MotionPlanner3DoF
from tools.Contour_detect import *

# ── CONFIGURATION PARAMETERS ──────────────────────────────────────────────────
YOLO_WEIGHTS        = 'yolo_weights/best.pt'  # REVERTED: Use stable weights from V2
CALIB_FILE_PATH     = "Camera_Calibration/camera_calibration.npz"
MARKER_LENGTH       = 20.6
MARKER_IN_BASE      = np.array([245, -45, 113.6])
BASE_CALIBRATION_ID = 0
TRANSFORM_MATRIX    = np.array([[0, 1], [1.0, 0]])
SCREW_Z_OFFSET      = -10
ROI_X, ROI_Y, ROI_W, ROI_H = 627, 140, 277, 208  
CAMERA_INDEX        = 0
EXECUTE_PICK        = True

# PRE-ASSIGNED DROP BIN COORDINATES FOR EACH TYPE: (Base, Shoulder, Elbow)
SORTING_BIN_MAP = {
    "M6":  (-55, 10.0, -100.0),
    "M8":  (-45.0, 10, -80.0),
    "M10": (-35, 5.0, -30),
    "M12": (45, 20.0, -90.0),
    "M??": (45, 20.0, -90.0)  
}

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

def main():
    kinematics = Kinematics3DOF()
    model      = YOLO(YOLO_WEIGHTS)
    mtx, dist  = load_calibration(CALIB_FILE_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    planner = None
    if EXECUTE_PICK:
        planner = MotionPlanner3DoF(port='COM3', baud_rate=9600)
        planner.connect_robot()
        planner.robot.home_robot()

    print("[INFO] Production Multi-Screw Sorting Active. Press [SPACE] to capture & sort batch.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if mtx is not None and dist is not None:
            frame = cv2.undistort(frame, mtx, dist, None, mtx)

        # ── STATE TRACKING: BUSY CHECK ──
        # Check if background thread queue is actively moving physical axes
        is_busy = planner and planner.robot and not planner.robot._queue.empty()

        # Draw the ROI boundary reference rectangle on the visualization stream
        cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 255), 2)

        # Detect ArUco Position Anchor Plane Context
        if hasattr(cv2.aruco, 'ArucoDetector'):
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50))

        anchor_found = False
        u_anchor, v_anchor, pixel_scale = 0.0, 0.0, 1.0

        if ids is not None and BASE_CALIBRATION_ID in ids.flatten():
            idx = np.where(ids.flatten() == BASE_CALIBRATION_ID)[0][0]
            mc = corners[idx][0]
            u_anchor, v_anchor = float(np.mean(mc[:, 0])), float(np.mean(mc[:, 1]))
            pixel_scale = MARKER_LENGTH / np.linalg.norm(mc[0] - mc[1])
            anchor_found = True
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Current display tracking tasks array
        current_frame_tasks = []

        # Only process frames/YOLO if the arm is completely finished moving
        if anchor_found and not is_busy:
            # Crop to specified ROI bounding zone matrix
            roi_frame = frame[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
            results = model(roi_frame, conf=0.4, verbose=False)

            for result in results:
                for box in result.boxes:
                    # Translate coordinates local to the ROI frame back to full frame space
                    rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = rx1 + ROI_X, ry1 + ROI_Y, rx2 + ROI_X, ry2 + ROI_Y
                    u_screw, v_screw = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    angles, robot_xyz = calculate_pick_profile(
                        u_anchor, v_anchor, u_screw, v_screw, pixel_scale, kinematics
                    )

                    if angles is not None:
                        # Extract Contour Verification on full frame crop
                        pad = 10
                        cropped_img = frame[max(0, int(y1)-pad):min(frame.shape[0], int(y2)+pad), 
                                            max(0, int(x1)-pad):min(frame.shape[1], int(x2)+pad)]
                        polygon = extract_screw_polygon(cropped_img)
                        size_str = "M??"
                        
                        if polygon is not None:
                            geom = analyze_geometry(polygon, cropped_img.shape)
                            
                            # Direct assignment since the helper function already formats it as "MX"
                            size_str = get_standard_screw_size(geom[0]*pixel_scale, geom[1]*pixel_scale, geom[2]*pixel_scale)
                        else:
                            size_str = "M??"

                        # The dictionary fallback map will now match keys normally
                        drop_bin_pose = SORTING_BIN_MAP.get(size_str, SORTING_BIN_MAP["M??"])
                        current_frame_tasks.append((angles, drop_bin_pose, size_str, (x1, y1, x2, y2)))

                        # Draw green boundary box markers
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 120), 2)
                        cv2.putText(frame, size_str, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:
                        # Non-reachable out-of-bounds indicators
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        # Provide operational status messages matching V2 logic
        h, w = frame.shape[:2]
        if is_busy:
            cv2.putText(frame, "EXECUTING CYCLE... (VISION PAUSED)", (w // 2 - 220, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif len(current_frame_tasks) > 0:
            cv2.putText(frame, f"READY: [SPACE] to sort {len(current_frame_tasks)} screws", (w // 2 - 200, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        else:
            cv2.putText(frame, "WAITING FOR TARGET TARGETS", (w // 2 - 150, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("Robot Arm Live Bounded Sorting Core", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        # Execute Batch Sorting operation sequence on Spacebar click
        if key == ord(' ') and EXECUTE_PICK and planner:
            if is_busy:
                print("[WARNING] Robot is currently executing a motion block. Command ignored.")
                continue

            if len(current_frame_tasks) == 0:
                print("[INFO] No viable coordinates locked in framework.")
                continue

            print(f"\n[LIVE BATCH] Launching sorting sequences for {len(current_frame_tasks)} items...")
            
            # Re-map items array down to clean kinematic tuple lists: (pick_angles, drop_angles)
            motion_tasks = [(task[0], task[1]) for task in current_frame_tasks]
            
            # Dispatch to underlying thread queue handler
            planner.run_multi_screw_sorting_cycle(motion_tasks)

            # Flush the hardware camera buffer completely to get rid of outdated cached frames
            for _ in range(5):
                cap.read()

    cap.release()
    cv2.destroyAllWindows()
    if planner: planner.disconnect_robot()

if __name__ == '__main__':
    main()