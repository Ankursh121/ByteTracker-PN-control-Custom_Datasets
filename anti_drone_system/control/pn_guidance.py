import numpy as np
import logging

logger = logging.getLogger("AntiDroneSystem.Control")

class PNGuidanceController:
    def __init__(self, config):
        """
        Initializes the enhanced Proportional Navigation controller.
        """
        self.config = config
        self.N = config['guidance']['nav_constant']
        self.max_speed_xy = config['guidance']['max_speed_xy']
        self.max_speed_z = config['guidance']['max_speed_z']
        self.max_yaw_rate = config['guidance']['max_yaw_rate']
        self.closing_speed = config['guidance']['closing_speed']
        self.target_lock_tolerance = config['guidance']['target_lock_tolerance']
        self.min_distance_intercept = config['guidance']['min_distance_intercept']
        self.deadzone_x = config['guidance']['deadzone_x']
        self.deadzone_y = config['guidance']['deadzone_y']
        self.los_smoothing = config['guidance']['los_rate_smoothing']
        
        self.k_yaw = 1.5
        
        # Saved states for smoothing
        self.smoothed_los_rate_y = 0.0
        self.smoothed_los_rate_p = 0.0
        self.last_track_id = None

    def compute_commands(self, target_track, distance, smoothed_pos, smoothed_vel, img_w, img_h, dt):
        """
        Computes 3D velocity commands and yaw rate. Applies deadzones and smooths LOS rates.
        
        Parameters:
            target_track (STrack): The active track object from ByteTrack
            distance (float): Estimated range in meters
            smoothed_pos (ndarray): [X, Y, Z] estimated position in camera frame
            smoothed_vel (ndarray): [Vx, Vy, Vz] estimated velocity in camera frame (m/frame)
            img_w (int): Width of the camera image frame
            img_h (int): Height of the camera image frame
            dt (float): Time step in seconds
            
        Returns:
            cmd (dict): Commands and status variables
        """
        track_id = target_track.track_id
        cx, cy, _, _ = target_track.xywh
        
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        # 1. Apply pixel-space deadzones to prevent high-frequency oscillations (hunting)
        dx = cx - cx_center
        dy = cy - cy_center
        
        if abs(dx) < self.deadzone_x:
            dx = 0.0
        else:
            # Shift error to avoid step jump at deadzone edge
            dx = dx - np.sign(dx) * self.deadzone_x
            
        if abs(dy) < self.deadzone_y:
            dy = 0.0
        else:
            dy = dy - np.sign(dy) * self.deadzone_y

        # Re-project deadzone-compensated coordinates into camera frame coordinates
        focal_length = self.config['ranging']['focal_length']
        X = (dx) * distance / focal_length
        Y = (dy) * distance / focal_length
        Z = distance
        
        # Convert target velocity from meters/frame to meters/second
        vel_m_s = smoothed_vel / max(dt, 1e-3)
        Vx_s, Vy_s, Vz_s = vel_m_s

        # 2. Compute Line of Sight (LOS) angles
        theta_y = np.arctan2(X, Z)
        theta_p = np.arctan2(-Y, Z)  # Negate Y since image Y-down is physical down

        # 3. Compute analytical LOS rates
        # d/dt arctan(X/Z) = (Vx*Z - X*Vz) / (X^2 + Z^2)
        raw_los_rate_y = (Vx_s * Z - X * Vz_s) / (X**2 + Z**2 + 1e-6)
        # d/dt arctan(-Y/Z) = (-Vy*Z + Y*Vz) / (Y^2 + Z^2)
        raw_los_rate_p = (-Vy_s * Z + Y * Vz_s) / (Y**2 + Z**2 + 1e-6)

        # Handle filter resets if targeting a new track ID
        if self.last_track_id != track_id:
            self.smoothed_los_rate_y = raw_los_rate_y
            self.smoothed_los_rate_p = raw_los_rate_p
            self.last_track_id = track_id

        # Smooth the LOS rates using Exponential Moving Average
        self.smoothed_los_rate_y = self.los_smoothing * raw_los_rate_y + (1.0 - self.los_smoothing) * self.smoothed_los_rate_y
        self.smoothed_los_rate_p = self.los_smoothing * raw_los_rate_p + (1.0 - self.los_smoothing) * self.smoothed_los_rate_p

        # 4. Vector Proportional Navigation Guidance:
        # Line-of-sight unit vector
        u_los = np.array([X, Y, Z]) / max(np.sqrt(X**2 + Y**2 + Z**2), 1e-6)
        
        # Target lateral velocity (cross-range velocity vector)
        target_vel_vec = np.array([Vx_s, Vy_s, Vz_s])
        target_vel_parallel = np.dot(target_vel_vec, u_los) * u_los
        target_vel_lateral = target_vel_vec - target_vel_parallel
        
        # Intercept velocity command in camera coordinate frame
        current_closing_speed = self.closing_speed
        if Z < self.min_distance_intercept:
            # Linear deceleration as the interceptor matches the target's position
            current_closing_speed = self.closing_speed * (Z / self.min_distance_intercept)
            
        vel_cmd_cam = current_closing_speed * u_los + self.N * target_vel_lateral
        
        # Map camera coordinates to drone body coordinates:
        # - vx_body = Z_cam (forward)
        # - vy_body = X_cam (right)
        # - vz_body = -Y_cam (up)
        vx_cmd = vel_cmd_cam[2]
        vy_cmd = vel_cmd_cam[0]
        vz_cmd = -vel_cmd_cam[1]

        # 5. Yaw Command: steer to point nose towards the target
        yaw_rate = self.k_yaw * theta_y
        
        # Apply saturation limits to command outputs
        vx_cmd = np.clip(vx_cmd, -self.max_speed_xy, self.max_speed_xy)
        vy_cmd = np.clip(vy_cmd, -self.max_speed_xy, self.max_speed_xy)
        vz_cmd = np.clip(vz_cmd, -self.max_speed_z, self.max_speed_z)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Check alignment (lock status)
        aligned = (abs(theta_y) < self.target_lock_tolerance) and (abs(theta_p) < self.target_lock_tolerance)

        return {
            'vx': float(vx_cmd),
            'vy': float(vy_cmd),
            'vz': float(vz_cmd),
            'yaw_rate': float(yaw_rate),
            'aligned': bool(aligned),
            'theta_y': float(theta_y),
            'theta_p': float(theta_p),
            'los_rate_y': float(self.smoothed_los_rate_y),
            'los_rate_p': float(self.smoothed_los_rate_p),
            'distance': float(Z)
        }

    def reset(self):
        """Resets saved smoothing history."""
        self.smoothed_los_rate_y = 0.0
        self.smoothed_los_rate_p = 0.0
        self.last_track_id = None
