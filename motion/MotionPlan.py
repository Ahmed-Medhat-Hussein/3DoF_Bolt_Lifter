import time

try:
    from motion.RobotArmController import RobotArmController
except ImportError:
    from RobotArmController import RobotArmController  

class MotionPlanner3DoF:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        if port == 'MOCK':
            self.robot = None
        else:
            self.robot = RobotArmController(port, baud_rate)
            
        # Standard structural mechanical home pose
        self.home_pose = {"base": 0.0, "shoulder": 60.0, "elbow": -125.0}

    def connect_robot(self):
        if self.robot is None: return True
        return self.robot.connect()

    def disconnect_robot(self):
        if self.robot is not None:
            self.robot.disconnect()

    def generate_place_and_lift(self, b, s, e):
        """Moves to bin, pauses 0.5s, turns relay OFF, and safe lifts shoulder +10 deg."""
        if self.robot is None:
            return [
                ('base', b), ('elbow', e), ('shoulder', s),
                ('delay', 0.5),
                ('relay', False),
                ('shoulder', s + 10.0)
            ]
        return [
            (self.robot.move_base, (b,)),
            (self.robot.move_elbow, (e,)),
            (self.robot.move_shoulder, (s,)),
            (self.robot.queue_delay, (0.5,)),
            (self.robot.set_relay, (False,)),
            (self.robot.move_shoulder, (s + 10.0,))
        ]

    def generate_home(self):
        """Moves safely back to structural Home alignment."""
        if self.robot is None:
            return [
                ('shoulder', self.home_pose["shoulder"]),
                ('elbow', self.home_pose["elbow"]),
                ('base', self.home_pose["base"])
            ]
        return [
                (self.robot.home_robot, ())
        ]
    

    def generate_pick_and_lift(self, b, s, e):
        """Moves directly to target, switches relay ON, pauses 2.0s to secure grip, and lifts to safe absolute clearance."""
        if self.robot is None:
            return [
                ('base', b), ('elbow', e), ('shoulder', s),
                ('relay', True),
                ('delay', 2.0),
                ('shoulder', 10.0) # Absolute clear height matching V2
            ]
        return [
            (self.robot.move_base, (b,)),
            (self.robot.move_elbow, (e,)),
            (self.robot.move_shoulder, (s,)),
            (self.robot.set_relay, (True,)),
            (self.robot.queue_delay, (2.0,)), # Match V2 secure capture delay
            (self.robot.move_shoulder, (10.0,)) # Match V2 absolute high clearance lift
        ]

    def run_multi_screw_sorting_cycle(self, compiled_screw_tasks):
        """Loops through target screw tasks. Safely homes between iterations to clear backlog step errors."""
        master_queue = []
        
        if not compiled_screw_tasks or compiled_screw_tasks[0] is None:
            print("[PLANNER] No task targets provided. Enqueueing safe Home routing.")
            return self.robot.queue_sequence(self.generate_home()) if self.robot else []

        print(f"[PLANNER] Merging {len(compiled_screw_tasks)} sorted cycle sequences into hardware thread...")
        
        for idx, (pick_joints, drop_joints) in enumerate(compiled_screw_tasks):
            p_b, p_s, p_e = pick_joints
            d_b, d_s, d_e = drop_joints
            
            # Step 1: Pick up screw and raise to clear zone
            master_queue.extend(self.generate_pick_and_lift(p_b, p_s, p_e))
            # Step 2: Drop screw off inside its pre-assigned M-Size bin location
            master_queue.extend(self.generate_place_and_lift(d_b, d_s, d_e))
            print(f" -> Pick Angles (IK): {pick_joints}")
            print(f" -> Drop Angles (Map): {drop_joints}")

            # CRITICAL FIX: Run true calibration homing between multiple targets to reset step drift
            if self.robot is not None:
                master_queue.append((self.robot.home_robot, ()))
                master_queue.append((self.robot.set_relay, (False,)))

        if self.robot is not None:
            self.robot.queue_sequence(master_queue)
            
        return master_queue