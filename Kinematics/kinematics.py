# kinematics.py
import numpy as np
from Kinematics import RobotConfig as cfg

class Kinematics3DOF:
    def dh_transformation_matrix(self, theta, d, a, alpha):
        """Calculates the transformation matrix for a single DH row."""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def forward_kinematics(self, theta1, theta2, theta3):
        """Calculates XYZ positions of all joints based on angles."""
        dh_table = cfg.get_dh_table(theta1, theta2, theta3)
        
        # Identity matrix for the base frame
        T0 = np.eye(4)
        joint_positions = [T0[:3, 3]] # Start with base at (0,0,0)
        
        T_current = T0
        for i in range(3):
            T_next = self.dh_transformation_matrix(*dh_table[i])
            T_current = np.dot(T_current, T_next)
            joint_positions.append(T_current[:3, 3])
        

            
        # Returns coordinates of: Base, Shoulder, Elbow, End-Effector, Camera
        return np.array(joint_positions)

    def is_within_limits(self, t1, t2, t3):
        """Checks if calculated angles are physically possible."""
        limits = cfg.JOINT_LIMITS
        if not (limits['theta1'][0] <= t1 <= limits['theta1'][1]): return False
        if not (limits['theta2'][0] <= t2 <= limits['theta2'][1]): return False
        if not (limits['theta3'][0] <= t3 <= limits['theta3'][1]): return False
        return True

    def inverse_kinematics(self, x, y, z):
        """
        Calculates joint angles to reach a specific X, Y, Z coordinate.
        Returns (theta1, theta2, theta3) in degrees, or None if unreachable.
        """
        try:
            theta1 = np.arctan2(y, x)
            
            r = np.sqrt(x**2 + y**2)
            z_offset = z - cfg.L1
            
            # Distance from shoulder joint to end-effector
            hypotenuse = np.sqrt(r**2 + z_offset**2)
            
            # Check if target is out of physical reach
            if hypotenuse > (cfg.L2 + cfg.L3):
                print("Target is too far!")
                return None
                
            cos_theta3 = (hypotenuse**2 - cfg.L2**2 - cfg.L3**2) / (2 * cfg.L2 * cfg.L3)

            # Clip to prevent floating point errors causing acos(1.0000000001)
            cos_theta3 = np.clip(cos_theta3, -1.0, 1.0) 
            
            # We take the negative angle for an "elbow up" configuration 
            theta3 = -np.arccos(cos_theta3) 
            
            alpha = np.arctan2(z_offset, r)
            beta = np.arctan2(cfg.L3 * np.sin(np.abs(theta3)), cfg.L2 + cfg.L3 * np.cos(theta3))
            theta2 = alpha + beta

            # Convert to degrees
            t1_deg = np.degrees(theta1)
            t2_deg = np.degrees(theta2)
            t3_deg = np.degrees(theta3)

            if self.is_within_limits(t1_deg, t2_deg, t3_deg):
                return (t1_deg, t2_deg, t3_deg)
            else:
                print("Target reachable, but violates joint limits!")
                return None

        except Exception as e:
            print(f"IK Math Error (Singularity): {e}")
            return None