import numpy as np
import logging

logger = logging.getLogger("AntiDroneSystem.Follow")

class FollowTargetController:
    def __init__(self, config):
        """
        Initializes the dedicated Follow Target controller with Visual Servoing and PN corrections.
        """
        self.config = config
        
        # Load configurations
        self.desired_follow_distance = config['guidance'].get('desired_follow_distance', 5.0)
        self.max_forward_speed = config['guidance'].get('max_forward_speed', 5.0)
        self.max_yaw_rate = config['guidance'].get('max_yaw_rate', 0.8)
        self.max_climb_rate = config['guidance'].get('max_climb_rate', 3.0)
        self.deadzone_x = config['guidance'].get('deadzone_x', 25.0)
        self.deadzone_y = config['guidance'].get('deadzone_y', 20.0)
        self.target_lock_tolerance = config['guidance'].get('target_lock_tolerance', 0.08)
        self.los_smoothing = config['guidance'].get('los_rate_smoothing', 0.3)
        self.N = config['guidance'].get('nav_constant', 4.0)

        # Control gains for visual servoing
        self.gain_yaw = 0.003       # rad/s per pixel error
        self.gain_vertical = 0.008  # m/s per pixel error (OpenCV Y-down)
        self.gain_forward = 0.8     # m/s per meter of distance error
        self.gain_lateral_correction = 0.002 # lateral m/s correction per pixel error

        # Smoothing states
        self.smooth_alpha = 0.25     # First-order command smoothing factor (prevents aggressive movements)
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0
        
        # PN state history
        self.smoothed_los_rate_y = 0.0
        self.smoothed_los_rate_p = 0.0
        self.last_track_id = None

    def compute_commands(self, target_track, distance, smoothed_pos, smoothed_vel, img_w, img_h, dt):
        """
        Computes follow-target commands combining visual servoing and PN guidance.
        Keep target centered, maintain configured chase distance, adjust yaw, forward velocity, and altitude.
        """
        track_id = target_track.track_id
        cx, cy, w_box, h_box = target_track.xywh
        
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        # 1. Image-center offset calculation
        error_x = cx - cx_center
        error_y = cy - cy_center
        
        # 2. Configurable deadzone filtering around image center to avoid oscillation
        if abs(error_x) < self.deadzone_x:
            error_x_dz = 0.0
        else:
            error_x_dz = error_x - np.sign(error_x) * self.deadzone_x
            
        if abs(error_y) < self.deadzone_y:
            error_y_dz = 0.0
        else:
            error_y_dz = error_y - np.sign(error_y) * self.deadzone_y

        # Re-project deadzone-compensated coordinates
        focal_length = self.config['ranging']['focal_length']
        X = (error_x_dz) * distance / focal_length
        Y = (error_y_dz) * distance / focal_length
        Z = distance
        
        # 3. Horizontal tracking (yaw rate)
        # Left of image center -> yaw left (negative); right -> yaw right (positive)
        raw_yaw_rate = self.gain_yaw * error_x_dz
        
        # 4. Vertical tracking (vertical velocity vz)
        # Upward in frame -> increase altitude (positive vz); downward -> decrease (negative vz)
        raw_vz = -self.gain_vertical * error_y_dz
        
        # 5. Distance tracking (forward velocity vx)
        # Farther than desired follow distance -> move forward (positive vx); closer -> move backward (negative vx)
        distance_error = distance - self.desired_follow_distance
        raw_vx = self.gain_forward * distance_error

        # 6. Lateral target drift tracking (lateral velocity vy)
        raw_vy = self.gain_lateral_correction * error_x_dz * (distance / 5.0)

        # 7. Enhanced PN Guidance Lateral/Target-Velocity Compensation
        # Incorporate PN guidance commands if target is moving to keep it centered during maneuvers
        if dt > 0.001 and smoothed_vel is not None:
            vel_m_s = smoothed_vel / dt
            Vx_s, Vy_s, Vz_s = vel_m_s

            u_los = np.array([X, Y, Z]) / max(np.sqrt(X**2 + Y**2 + Z**2), 1e-6)
            target_vel_vec = np.array([Vx_s, Vy_s, Vz_s])
            target_vel_parallel = np.dot(target_vel_vec, u_los) * u_los
            target_vel_lateral = target_vel_vec - target_vel_parallel
            
            # Target lateral velocity commands from PN
            pn_lateral_cam = self.N * target_vel_lateral
            
            # Add PN-derived lateral correction into our body-frame command
            # Camera coordinates to body: vx=Z_cam, vy=X_cam, vz=-Y_cam
            raw_vy += pn_lateral_cam[0]
            raw_vz += -pn_lateral_cam[1]

        # 8. Command smoothing to prevent aggressive movements
        vx = self.smooth_alpha * raw_vx + (1 - self.smooth_alpha) * self.last_vx
        vy = self.smooth_alpha * raw_vy + (1 - self.smooth_alpha) * self.last_vy
        vz = self.smooth_alpha * raw_vz + (1 - self.smooth_alpha) * self.last_vz
        yaw_rate = self.smooth_alpha * raw_yaw_rate + (1 - self.smooth_alpha) * self.last_yaw_rate
        
        # Save command history for next step
        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz
        self.last_yaw_rate = yaw_rate

        # 9. Safety Clamping (Velocities and rate limits)
        vx = np.clip(vx, -self.max_forward_speed, self.max_forward_speed)
        vy = np.clip(vy, -self.max_forward_speed, self.max_forward_speed)
        vz = np.clip(vz, -self.max_climb_rate, self.max_climb_rate)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Check target alignment (aligned within tolerance)
        theta_y = np.arctan2(X, Z)
        theta_p = np.arctan2(-Y, Z)
        aligned = (abs(theta_y) < self.target_lock_tolerance) and (abs(theta_p) < self.target_lock_tolerance)

        return {
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'yaw_rate': float(yaw_rate),
            'aligned': bool(aligned),
            'theta_y': float(theta_y),
            'theta_p': float(theta_p),
            'distance': float(distance),
            'desired_distance': float(self.desired_follow_distance)
        }

    def reset(self):
        """Resets controller states."""
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0
        self.smoothed_los_rate_y = 0.0
        self.smoothed_los_rate_p = 0.0
        self.last_track_id = None
