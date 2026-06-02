import numpy as np

# --- Link Lengths (in mm) ---
L1 = 270 # Actual is 239mm + Ground elevation of the base
L2 = 199.3  
L3 = 130    

# --- Joint Limits (in degrees) ---
JOINT_LIMITS = {
    'theta1': (-180, 180),
    'theta2': (-25, 110),  
    'theta3': (-135, 20)  
}

def get_dh_table(theta1, theta2, theta3):
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)
    
    return [
        [t1, L1, 0,  np.pi/2], 
        [t2, 0,  L2, 0],       
        [t3, 0,  L3, 0]        
    ]