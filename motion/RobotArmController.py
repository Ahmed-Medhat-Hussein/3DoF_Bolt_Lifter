import serial
import time
import threading
import queue

class RobotArmController:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._worker_thread = None

    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            # Wait for the Arduino to reboot and send the "READY" message
            print(f"Connecting to {self.port}...")
            self._wait_for_startup()
            return True
        except serial.SerialException as e:
            print(f"Connection Error: {e}")
            return False

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected.")

    def _wait_for_startup(self):
        """Waits for the Arduino to signal it has booted."""
        while True:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line == "READY":
                    print("Robot is READY.")
                    break
                

    def stalling_timer(self, duration):
        """Sets the busy flag for a set time using a background thread."""
        def _wait():
            self.is_busy = True
            print(f"[ROBOT] Starting {duration}s pause...")
            time.sleep(duration)
            self.is_busy = False
            print("[ROBOT] Pause complete.")
        
        threading.Thread(target=_wait, daemon=True).start()

    
    def queue_sequence(self, sequence):
        """Accepts a list of tuples: (function, arguments_tuple) or (function,)"""
        for item in sequence:
            if len(item) == 1:
                self._queue.put((item[0], ()))
            else:
                self._queue.put((item[0], item[1]))
                
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
                self._worker_thread.start()

    def _process_queue(self):
        """Internal worker that runs actions sequentially without blocking the main loop."""
        while not self._queue.empty():
            func, args = self._queue.get()
            try:
                func(*args)
            except Exception as e:
                print(f"[QUEUE ERROR] Failed executing {func.__name__}: {e}")
            self._queue.task_done()

    def queue_delay(self, duration):
        """A helper delay that only halts the background queue execution thread."""
        time.sleep(duration)

    def _send_and_wait(self, command_char, value=None):
        if not self.serial_conn or not self.serial_conn.is_open:
            return "Error: Port closed."
        
        # 1. Clear the buffer so we don't read an old "DONE"
        self.serial_conn.reset_input_buffer()

        # 2. Format and send the command ONCE
        cmd_str = f"{command_char}{float(value):.2f}\n" if value is not None else f"{command_char}\n"
        self.serial_conn.write(cmd_str.encode('utf-8'))
        
        # 3. Active Polling: This blocks the script until the Arduino says "DONE"
        response_log = []
        while True:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line == "DONE":
                    break
                elif line:
                    response_log.append(line)
                    
        return " | ".join(response_log)

    # --- Synchronous Control Functions ---
    def move_base(self, angle):
        return self._send_and_wait('1', float(angle))

    def move_shoulder(self, angle):
        return self._send_and_wait('2', float(angle))

    def move_elbow(self, angle):
        return self._send_and_wait('3', float(angle))

    def home_robot(self):
        return self._send_and_wait('H')

    def set_relay(self, state):
        return self._send_and_wait('R', 1 if state else 0)

# ==========================================
# Example: Nested / Sequential Commands
# ==========================================
if __name__ == "__main__":
    robot = RobotArmController(port='COM3', baud_rate=9600) 
    
    if robot.connect():
        # Because of the state updater, these will execute flawlessly in sequence.
        # Python will not send the next command until the previous physical motion ends.
        
        print("Homing...")
        print(robot.home_robot())
        
        print("Moving Base to 90...")
        print(robot.move_base(90.0))
        
        print("Moving Shoulder to 60...")
        print(robot.move_shoulder(60.0))
        
        print("Activating Relay...")
        print(robot.set_relay(True))
        
        print("Sequence Complete.")
        robot.disconnect()