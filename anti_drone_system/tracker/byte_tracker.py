import numpy as np
import logging
from .kalman_tracker import STrack, KalmanFilter, TrackState
from .association import iou_distance, linear_assignment

logger = logging.getLogger("AntiDroneSystem.Tracker")

class ByteTracker:
    def __init__(self, config):
        """
        Initializes the ByteTrack tracker with configuration settings.
        """
        self.config = config
        self.track_thresh = config['tracker']['track_threshold']
        self.match_thresh = config['tracker']['match_threshold']
        self.track_buffer = config['tracker']['track_buffer']
        self.min_box_area = config['tracker']['min_box_area']
        
        self.kalman_filter = KalmanFilter()
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        
        # Reset ID counter when initialized
        STrack.reset_id_counter()

    def update(self, detections_list):
        """
        Updates the tracker with detections from the current frame.
        detections_list: list of [x1, y1, x2, y2, score, class_id]
        Returns:
            list of active STrack objects
        """
        self.frame_id += 1
        
        # 1. Parse and filter detections by size/area
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        detections_high = []
        detections_low = []

        for det in detections_list:
            x1, y1, x2, y2, score, cls_id = det
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area < self.min_box_area:
                continue
                
            tracklet = STrack([x1, y1, w, h], score, cls_id)
            
            # Split detections by score threshold
            if score >= self.track_thresh:
                detections_high.append(tracklet)
            else:
                detections_low.append(tracklet)

        # 2. Predict the new locations of existing tracked and lost tracks using Kalman filter
        unconfirmed_stracks = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)

        # Merge tracked and lost tracks for joint prediction
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Run prediction step
        for track in strack_pool:
            track.predict(self.kalman_filter)

        # 3. First Association: Match high-confidence detections with the predicted track pool
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_det_high = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            
            if track.state == TrackState.Tracked:
                track.update(self.kalman_filter, det, self.frame_id)
                activated_stracks.append(track)
            else:
                # Re-activate lost track
                track.re_activate(self.kalman_filter, det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 4. Second Association: Match remaining tracked tracks (from first stage) with low-confidence detections
        # Note: We only match tracks that are currently in the 'Tracked' state.
        # Unmatched tracks from the first stage that were 'Lost' are not matched in this step.
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = iou_distance(r_tracked_stracks, detections_low)
        matches, u_track_second, u_det_low = linear_assignment(dists, thresh=0.5)  # Strict IoU threshold for low confidence

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            track.update(self.kalman_filter, det, self.frame_id)
            activated_stracks.append(track)

        # Unmatched tracks from the second stage association are marked lost
        for itracked in u_track_second:
            track = r_tracked_stracks[itracked]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 5. Deal with unconfirmed tracks (new tracks from last frame that were not confirmed yet)
        # Match them with remaining unmatched high-confidence detections
        r_unconfirmed = [t for t in unconfirmed_stracks if t.state == TrackState.New or t.state == TrackState.Tracked]
        u_det_high_stracks = [detections_high[i] for i in u_det_high]
        
        dists = iou_distance(r_unconfirmed, u_det_high_stracks)
        matches, u_unconfirmed, u_det_new = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = r_unconfirmed[itracked]
            det = u_det_high_stracks[idet]
            track.update(self.kalman_filter, det, self.frame_id)
            activated_stracks.append(track)

        for itracked in u_unconfirmed:
            track = r_unconfirmed[itracked]
            track.mark_removed()
            removed_stracks.append(track)

        # 6. Initialize new tracks from unmatched high-confidence detections
        for idet in u_det_new:
            det = u_det_high_stracks[idet]
            # Immediately activate track
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)

        # 7. Handle lost tracks from the pool that were not matched at all
        for it in u_track:
            track = strack_pool[it]
            if track.state == TrackState.Tracked:
                # If they were tracked but not matched in either association stage, mark lost
                track.mark_lost()
                lost_stracks.append(track)
            elif track.state == TrackState.Lost:
                # If they were already lost and still not matched, they remain lost
                pass

        # 8. Clean up lost tracks that exceeded track_buffer
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # 9. Update lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        
        # Filter lost tracks list
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        
        # Keep track list of active tracks (which we output for visualization/guidance)
        # We output tracks that are currently tracked (active)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """
        Merges two lists of tracks based on ID.
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = True
            res.append(t)
        for t in tlistb:
            if t.track_id not in exists:
                exists[t.track_id] = True
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """
        Subtracts tracks in tlistb from tlista based on ID.
        """
        res = {}
        for t in tlista:
            res[t.track_id] = t
        for t in tlistb:
            if t.track_id in res:
                del res[t.track_id]
        return list(res.values())
