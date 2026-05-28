import numpy as np

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    _count = 0

    def __init__(self, tlwh, score, class_id=0):
        # Bounding box in top-left-width-height format
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.time_since_update = 0
        
        # Kalman Filter state: mean (8,) and covariance (8, 8)
        self.mean = None
        self.covariance = None

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id_counter():
        STrack._count = 0

    def predict(self, kalman_filter):
        """
        Predicts the next state of the track using the Kalman filter.
        """
        mean, covariance = kalman_filter.predict(self.mean, self.covariance)
        self.mean = mean
        self.covariance = covariance
        self.time_since_update += 1

    def activate(self, kalman_filter, frame_id):
        """
        Activates a new track.
        """
        self.track_id = STrack.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.to_xyah(self._tlwh))
        
        self.time_since_update = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.is_activated = True

    def re_activate(self, kalman_filter, new_track, frame_id, new_id=False):
        """
        Re-activates a lost track with a new detection.
        """
        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, self.to_xyah(new_track._tlwh)
        )
        self.time_since_update = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = STrack.next_id()

    def update(self, kalman_filter, new_track, frame_id):
        """
        Updates the track with a new matched detection.
        """
        self.frame_id = frame_id
        self.time_since_update = 0
        self.tracklet_len += 1
        
        new_xyah = self.to_xyah(new_track._tlwh)
        self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, new_xyah)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        """
        Get current position in bounding box format [top left x, top left y, width, height].
        If Kalman filter is initialized, returns filtered estimate.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect_ratio * height -> width
        ret[0] -= ret[2] / 2
        ret[1] -= ret[3] / 2
        return ret

    @property
    def tlbr(self):
        """
        Get current position in [top left x, top left y, bottom right x, bottom right y] format.
        """
        ret = self.tlwh
        ret[2] += ret[0]
        ret[3] += ret[1]
        return ret

    @property
    def xywh(self):
        """
        Get current position in [center x, center y, width, height] format.
        """
        ret = self.tlwh
        ret[0] += ret[2] / 2
        ret[1] += ret[3] / 2
        return ret

    @staticmethod
    def to_xyah(tlwh):
        """
        Convert bounding box to format [center x, center y, aspect ratio (w/h), height].
        """
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[0] += ret[2] / 2
        ret[1] += ret[3] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class KalmanFilter:
    """
    A simple 8-dimensional Kalman filter for tracking bounding boxes in image space.
    State representation: [x, y, a, h, vx, vy, va, vh]
    """
    def __init__(self):
        ndim = 4
        dt = 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Standard deviations for process and measurement noise
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """
        Create track from first measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Covariance depends on box height (scale-invariant initialization)
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h
        ]
        motion_cov = np.diag(np.square(std))

        # x' = F x
        mean = np.dot(self._motion_mat, mean)
        # P' = F P F^T + Q
        covariance = np.dot(self._motion_mat, np.dot(covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h
        ]
        measurement_cov = np.diag(np.square(std))

        # Innovation (residual)
        # y = z - H x
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.dot(self._update_mat, np.dot(covariance, self._update_mat.T)) + measurement_cov

        # Kalman gain: K = P H^T (H P H^T + R)^-1
        # Using Cholesky/Solve to be numerically stable
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(chol_factor.T, np.linalg.solve(chol_factor, np.dot(self._update_mat, covariance))).T

        # Update mean: x = x + K y
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        
        # Update covariance: P = (I - K H) P
        new_covariance = covariance - np.dot(kalman_gain, np.dot(projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
