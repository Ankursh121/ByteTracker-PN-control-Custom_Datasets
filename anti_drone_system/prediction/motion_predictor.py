import numpy as np
import logging

logger = logging.getLogger("AntiDroneSystem.Prediction")

class MotionPredictor:
    def __init__(self, config):
        """
        Initializes the 3D Kalman Filter Motion Predictor.
        """
        self.config = config
        self.focal_length = config['ranging']['focal_length']
        
        # State: [X, Y, Z, Vx, Vy, Vz]
        self.state = None
        self.covariance = None
        self.last_track_id = None
        
        # Time step (assumed 1.0 frame step for simplicity, scaled by FPS in control)
        self.dt = 1.0
        
        # Setup Kalman Filter matrices
        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, 3 + i] = self.dt
            
        self.H = np.eye(3, 6, dtype=np.float32)
        
        # Process noise covariance Q
        self.Q = np.eye(6, dtype=np.float32) * 0.05
        self.Q[3:, 3:] = np.eye(3, dtype=np.float32) * 0.2  # Higher noise for velocity changes
        
        # Measurement noise covariance R
        self.R = np.diag([0.1, 0.1, 0.5]).astype(np.float32)  # Range (Z) has higher noise

    def update(self, track_id, cx, cy, distance, img_w, img_h):
        """
        Updates the 3D position filter with a new observation.
        
        Parameters:
            track_id (int): Unique target track ID
            cx (float): Target center X in pixels
            cy (float): Target center Y in pixels
            distance (float): Target distance in meters
            img_w (float): Frame width in pixels
            img_h (float): Frame height in pixels
            
        Returns:
            smoothed_pos (ndarray): [X, Y, Z] in camera frame
            smoothed_vel (ndarray): [Vx, Vy, Vz] in camera frame (m/frame)
        """
        # 1. Project pixel coordinates to 3D camera coordinates
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        x_3d = (cx - cx_center) * distance / self.focal_length
        y_3d = (cy - cy_center) * distance / self.focal_length
        z_3d = distance
        
        measurement = np.array([x_3d, y_3d, z_3d], dtype=np.float32)
        
        # If tracking a new target, re-initialize state
        if self.state is None or self.last_track_id != track_id:
            logger.info(f"Initializing 3D prediction filter for track ID: {track_id}")
            self.state = np.zeros(6, dtype=np.float32)
            self.state[:3] = measurement
            self.covariance = np.eye(6, dtype=np.float32) * 1.0
            self.last_track_id = track_id
            return self.state[:3], self.state[3:]
            
        # 2. Predict Step
        # x' = F x
        # P' = F P F^T + Q
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(self.F, np.dot(self.covariance, self.F.T)) + self.Q
        
        # 3. Update Step (Correction)
        # y = z - H x
        # S = H P H^T + R
        # K = P H^T S^-1
        # x = x + K y
        # P = (I - K H) P
        projected_meas = np.dot(self.H, self.state)
        residual = measurement - projected_meas
        
        S = np.dot(self.H, np.dot(self.covariance, self.H.T)) + self.R
        K = np.dot(self.covariance, np.dot(self.H.T, np.linalg.inv(S)))
        
        self.state = self.state + np.dot(K, residual)
        self.covariance = self.covariance - np.dot(K, np.dot(self.H, self.covariance))
        
        return self.state[:3], self.state[3:]

    def predict_future(self, steps=5):
        """
        Predicts the 3D position of the target 'steps' frames into the future.
        
        Returns:
            predicted_3d (ndarray): [X, Y, Z] predicted positions
        """
        if self.state is None:
            return None
            
        pos = self.state[:3]
        vel = self.state[3:]
        
        # Simple linear projection: P_future = P + steps * V
        return pos + steps * vel

    def project_to_image(self, pos_3d, img_w, img_h):
        """
        Projects a 3D camera coordinate point back to 2D image coordinates.
        
        Parameters:
            pos_3d (ndarray): [X, Y, Z] position
            img_w (float): Frame width
            img_h (float): Frame height
            
        Returns:
            px, py (float, float): Pixel coordinates on screen
        """
        if pos_3d is None:
            return None
            
        X, Y, Z = pos_3d
        if Z <= 0.1:
            return None
            
        cx_center = img_w / 2.0
        cy_center = img_h / 2.0
        
        px = (X * self.focal_length / Z) + cx_center
        py = (Y * self.focal_length / Z) + cy_center
        
        return int(px), int(py)

    def reset(self):
        """
        Resets the predictor state.
        """
        self.state = None
        self.covariance = None
        self.last_track_id = None
