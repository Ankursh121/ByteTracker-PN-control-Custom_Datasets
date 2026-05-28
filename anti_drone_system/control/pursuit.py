import numpy as np
import logging

logger = logging.getLogger("AntiDroneSystem.Pursuit")

class DirectPursuitController:
    def __init__(self, config):
        """
        Initializes the Direct Pursuit (rule-based) controller.
        """
        self.config = config
        self.max_speed_xy = config['guidance']['max_speed_xy']
        self.max_speed_z = config['guidance']['max_speed_z']
        self.max_yaw_rate = config['guidance']['max_yaw_rate']
        self.closing_speed = config['guidance']['closing_speed']
        self.target_lock_tolerance = config['guidance']['target_lock_tolerance']
        self.min_distance_intercept = config['guidance']['min_distance_intercept']
        self.deadzone_x = config['guidance']['deadzone_x']
        self.deadzone_y = config['guidance']['deadzone_y']
        
        # Pursuit gains
        self.gain_yaw = 0.005    # yaw rate per pixel error
        self.gain_vertical = 0.004  # vertical speed per pixel error
        self.gain_forward = 0.5   # forward speed scaling factor
        
        # Command smoothing state (first order filter)
        self.smooth_alpha = 0.3
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0

    def compute_commands(self, target_track, distance, img_w, img_h):
        """
        Computes velocity commands based on simple direct steering rules.
        
        Parameters:
            target_track (STrack): Active track from ByteTracker
            distance (float): Estimated range in meters
            img_w (int): Frame width
            img_h (int): Frame height
            
        Returns:
            cmd (dict): Velocity commands
        """
        cx, cy, _, _ = target_track.xywh
        
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        # Compute raw pixel offsets
        error_x = cx - cx_center
        error_y = cy - cy_center  # Positive is down
        
        # 1. Apply deadzones
        if abs(error_x) < self.deadzone_x:
            error_x = 0.0
            
        if abs(error_y) < self.deadzone_y:
            error_y = 0.0

        # 2. Rule-Based Target Following Control:
        # A. Yaw: target left -> yaw left, target right -> yaw right
        raw_yaw_rate = self.gain_yaw * error_x
        
        # B. Vertical Speed: target high -> move up, target low -> move down
        # OpenCV Y-down, so positive error_y is down, which commands negative vertical velocity
        raw_vz = -self.gain_vertical * error_y
        
        # C. Forward Speed: target far -> move forward, target close -> slow down
        if distance > self.min_distance_intercept:
            raw_vx = self.closing_speed
        else:
            # Linear deceleration as we close in
            raw_vx = self.closing_speed * (distance / self.min_distance_intercept)
            
        # D. Lateral Speed (drift correction to center target horizontally)
        # Command a small lateral velocity to help align
        raw_vy = 0.003 * error_x * distance

        # 3. Apply Command Smoothing (prevent sudden jumps)
        vx = self.smooth_alpha * raw_vx + (1 - self.smooth_alpha) * self.last_vx
        vy = self.smooth_alpha * raw_vy + (1 - self.smooth_alpha) * self.last_vy
        vz = self.smooth_alpha * raw_vz + (1 - self.smooth_alpha) * self.last_vz
        yaw_rate = self.smooth_alpha * raw_yaw_rate + (1 - self.smooth_alpha) * self.last_yaw_rate

        # Save states
        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz
        self.last_yaw_rate = yaw_rate

        # 4. Limit control outputs (Saturation)
        vx = np.clip(vx, -self.max_speed_xy, self.max_speed_xy)
        vy = np.clip(vy, -self.max_speed_xy, self.max_speed_xy)
        vz = np.clip(vz, -self.max_speed_z, self.max_speed_z)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Alignment criteria
        focal_length = self.config['ranging']['focal_length']
        theta_y = np.arctan2(error_x * distance / focal_length, distance)
        theta_p = np.arctan2(-error_y * distance / focal_length, distance)
        aligned = (abs(theta_y) < self.target_lock_tolerance) and (abs(theta_p) < self.target_lock_tolerance)

        return {
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'yaw_rate': float(yaw_rate),
            'aligned': bool(aligned),
            'theta_y': float(theta_y),
            'theta_p': float(theta_p),
            'distance': float(distance)
        }

    def reset(self):
        """Resets smoothing memory."""
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0
