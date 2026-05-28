import time
import numpy as np
import cv2
import logging

logger = logging.getLogger("AntiDroneSystem.Simulation")

class SITLSimulator:
    def __init__(self, config):
        """
        Initializes the SITL Simulator.
        """
        self.config = config
        self.focal_length = config['ranging']['focal_length']
        self.target_size = config['ranging']['drone_real_size']
        
        # Virtual target initial position in local NED (North, East, Down relative to home)
        # Place target 25 meters North, 0 meters East, and 0 meters Down (0m altitude) to be visible on startup
        self.target_pos = np.array([25.0, 0.0, 0.0], dtype=np.float32)
        
        # Target trajectory parameters
        self.start_time = time.time()
        self.trajectory_type = "circle"  # "circle", "hover", "sine"
        self.speed = 1.25                 # Speed of virtual target movement (m/s)
        self.radius = 8.0                # Radius of circle trajectory (m)
        self.center = np.array([25.0, 0.0, 0.0], dtype=np.float32)

    def update_target_position(self):
        """
        Updates the virtual moving target's position over time.
        """
        t = time.time() - self.start_time
        
        if self.trajectory_type == "circle":
            # Circular path in horizontal plane
            angle = (self.speed * t) / self.radius
            self.target_pos[0] = self.center[0] + self.radius * np.cos(angle)
            self.target_pos[1] = self.center[1] + self.radius * np.sin(angle)
            self.target_pos[2] = self.center[2] + 1.5 * np.sin(0.2 * t)  # Gentle altitude wave
        elif self.trajectory_type == "sine":
            # Linear back-and-forth movement
            self.target_pos[0] = self.center[0] + 15.0 * np.sin(0.1 * t)
            self.target_pos[1] = self.center[1]
            self.target_pos[2] = self.center[2]
        else:
            # Hover at fixed position
            self.target_pos = self.center.copy()

    def project_target_to_camera(self, interceptor_pos, interceptor_attitude, img_w, img_h):
        """
        Projects the virtual target NED position onto the interceptor camera plane.
        
        Parameters:
            interceptor_pos (ndarray): [North, East, Down] of the interceptor
            interceptor_attitude (ndarray): [roll, pitch, yaw] in radians
            img_w (int): Camera frame width
            img_h (int): Camera frame height
            
        Returns:
            detection: [x1, y1, x2, y2, confidence, class_id] if visible, else None
            relative_3d: [X_cam, Y_cam, Z_cam] target position relative to camera
        """
        # 1. Update virtual target path
        self.update_target_position()
        
        # 2. Compute relative position in local NED frame
        rel_ned = self.target_pos - interceptor_pos
        
        # 3. Rotate relative NED position to interceptor Body Frame
        # Attitude angles
        roll, pitch, yaw = interceptor_attitude
        
        # Rotation matrices (Local to Body / Yaw -> Pitch -> Roll)
        # R_yaw (Z-axis)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [cy, sy, 0],
            [-sy, cy, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # R_pitch (Y-axis)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_pitch = np.array([
            [cp, 0, -sp],
            [0, 1, 0],
            [sp, 0, cp]
        ], dtype=np.float32)
        
        # R_roll (X-axis)
        cr, sr = np.cos(roll), np.sin(roll)
        R_roll = np.array([
            [1, 0, 0],
            [0, cr, sr],
            [0, -sr, cr]
        ], dtype=np.float32)
        
        # Combined rotation: Local to Body
        R = np.dot(R_roll, np.dot(R_pitch, R_yaw))
        rel_body = np.dot(R, rel_ned)  # [x_forward, y_right, z_down]
        
        # 4. Map Body coordinates to OpenCV Camera Frame coordinates
        # - X_cam = y_right
        # - Y_cam = z_down
        # - Z_cam = x_forward
        X_cam = rel_body[1]
        Y_cam = rel_body[2]
        Z_cam = rel_body[0]
        
        relative_3d = np.array([X_cam, Y_cam, Z_cam])

        # 5. Check if target is behind the camera (not visible)
        if Z_cam <= 0.2:
            return None, relative_3d

        # 6. Project to 2D Pixel coordinates using camera pinhole model
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        px = (X_cam * self.focal_length / Z_cam) + cx_center
        py = (Y_cam * self.focal_length / Z_cam) + cy_center
        
        # Calculate bounding box size based on range and real drone physical size
        # box_size_px = focal_length * real_size / distance
        box_size = (self.focal_length * self.target_size) / Z_cam
        
        # Check if projected center is within the camera image boundaries (plus margin)
        margin = 150
        if (-margin <= px <= img_w + margin) and (-margin <= py <= img_h + margin):
            x1 = px - box_size / 2.0
            y1 = py - box_size / 2.0
            x2 = px + box_size / 2.0
            y2 = py + box_size / 2.0
            
            # Simulated detection list format [x1, y1, x2, y2, confidence, class_id]
            # Add a tiny bit of random noise to simulate real-world jitter
            noise = np.random.normal(0, 0.5, 4)
            detection = [
                float(x1 + noise[0]),
                float(y1 + noise[1]),
                float(x2 + noise[2]),
                float(y2 + noise[3]),
                float(np.clip(0.85 + np.random.normal(0, 0.02), 0.0, 1.0)),  # High confidence
                0  # Class 0: drone
            ]
            return detection, relative_3d
            
        return None, relative_3d

    def generate_virtual_frame(self, detection, img_w, img_h):
        """
        Generates a synthetic camera frame displaying the virtual target drone
        if no real camera feed is active.
        """
        # Create a deep space blue tactical HUD background
        frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        frame[:, :] = [20, 15, 10]  # Very dark blue-gray
        
        # Draw some grid lines to look like a digital flight interface
        grid_size = 80
        for x in range(0, img_w, grid_size):
            cv2.line(frame, (x, 0), (x, img_h), (35, 30, 25), 1)
        for y in range(0, img_h, grid_size):
            cv2.line(frame, (0, y), (img_w, y), (35, 30, 25), 1)
            
        # Draw the virtual drone target visually as a tactical crosshair or target marker
        if detection is not None:
            x1, y1, x2, y2, _, _ = detection
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            w = int(x2 - x1)
            
            # Draw synthetic drone shape
            color = (0, 0, 255)  # Red target
            cv2.circle(frame, (cx, cy), max(4, int(w * 0.15)), color, -1)
            # Draw quadcopter arms
            arm_len = max(8, int(w * 0.45))
            cv2.line(frame, (cx - arm_len, cy - arm_len), (cx + arm_len, cy + arm_len), (120, 120, 120), 2)
            cv2.line(frame, (cx - arm_len, cy + arm_len), (cx + arm_len, cy - arm_len), (120, 120, 120), 2)
            # Rotors
            cv2.circle(frame, (cx - arm_len, cy - arm_len), max(2, int(w * 0.1)), (200, 200, 200), 1)
            cv2.circle(frame, (cx + arm_len, cy - arm_len), max(2, int(w * 0.1)), (200, 200, 200), 1)
            cv2.circle(frame, (cx - arm_len, cy + arm_len), max(2, int(w * 0.1)), (200, 200, 200), 1)
            cv2.circle(frame, (cx + arm_len, cy + arm_len), max(2, int(w * 0.1)), (200, 200, 200), 1)
            
        return frame
