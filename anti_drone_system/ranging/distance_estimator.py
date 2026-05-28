import logging

logger = logging.getLogger("AntiDroneSystem.Ranging")

class DistanceEstimator:
    def __init__(self, config):
        """
        Initializes the range estimator.
        """
        self.config = config
        self.focal_length = config['ranging']['focal_length']
        self.drone_real_size = config['ranging']['drone_real_size']
        self.alpha = config['ranging']['filter_alpha']
        
        # Track historical smoothed distances by track ID
        self.history = {}

    def estimate(self, track_id, bbox_w, bbox_h):
        """
        Estimates the distance to the drone using the monocular bounding box size.
        Applies Exponential Moving Average (EMA) filtering per track.
        
        Parameters:
            track_id (int): Unique ID of the track
            bbox_w (float): Width of the bounding box in pixels
            bbox_h (float): Height of the bounding box in pixels
            
        Returns:
            smoothed_distance (float): Smoothed estimated distance in meters
        """
        # We use the maximum of width and height to represent the target's physical size
        # This is more robust against drone rotations/tilting compared to just width or height
        bbox_size = max(bbox_w, bbox_h)
        
        if bbox_size <= 0:
            logger.warning(f"Invalid bounding box size ({bbox_size}) for track {track_id}")
            return 0.0

        # Calculate raw distance: d = f * W / w_px
        raw_distance = (self.focal_length * self.drone_real_size) / bbox_size
        
        # Apply EMA filter
        if track_id in self.history:
            last_smoothed = self.history[track_id]
            smoothed_distance = self.alpha * raw_distance + (1.0 - self.alpha) * last_smoothed
        else:
            # First observation
            smoothed_distance = raw_distance
            
        self.history[track_id] = smoothed_distance
        return smoothed_distance

    def clean_track(self, track_id):
        """
        Removes history for a track when it is deleted.
        """
        if track_id in self.history:
            del self.history[track_id]
            
    def reset(self):
        """
        Clears all history.
        """
        self.history.clear()
