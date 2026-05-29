import numpy as np
import cv2

class BoundingBoxTracker:
    """
    Lucas-Kanade optical flow based tracker to maintain target locks 
    on drones during YOLO model dropouts or motion blur.
    """
    def __init__(self):
        self.box = None # [x1, y1, x2, y2] in pixels
        self.prev_gray = None
        self.features = None
        self.active = False
        self.last_conf = 0.0
        
    def init_tracker(self, frame_gray, box):
        x1, y1, x2, y2 = box
        h, w = frame_gray.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.active = False
            return
            
        # Focus features towards the center of the bounding box to avoid background noise/clutter
        bw = x2 - x1
        bh = y2 - y1
        pad_w = int(bw * 0.20) if bw > 15 else 0
        pad_h = int(bh * 0.20) if bh > 15 else 0
        
        # Create mask for features within target bounding box (inner region)
        mask = np.zeros_like(frame_gray)
        mask[y1+pad_h : y2-pad_h, x1+pad_w : x2-pad_w] = 255
        
        # Detect tracking features inside target box
        pts = cv2.goodFeaturesToTrack(
            frame_gray, 
            maxCorners=25, 
            qualityLevel=0.03, 
            minDistance=3, 
            mask=mask
        )
        
        if pts is not None and len(pts) > 0:
            self.box = [x1, y1, x2, y2]
            self.prev_gray = frame_gray.copy()
            self.features = pts.copy()
            self.active = True
        else:
            # Fallback: create a grid of points inside the box to track
            grid_y, grid_x = np.mgrid[y1+pad_h+2 : y2-pad_h-2 : 4, x1+pad_w+2 : x2-pad_w-2 : 4]
            pts = np.vstack((grid_x.flatten(), grid_y.flatten())).T.reshape(-1, 1, 2).astype(np.float32)
            if len(pts) > 0:
                self.box = [x1, y1, x2, y2]
                self.prev_gray = frame_gray.copy()
                self.features = pts
                self.active = True
            else:
                self.active = False
            
    def update_tracker(self, frame_gray):
        if not self.active or self.features is None or self.prev_gray is None:
            self.active = False
            return False, None
            
        # Calculate optical flow of tracking points (expanded window and level for fast motion)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, 
            frame_gray, 
            self.features, 
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if next_pts is not None and status is not None:
            good_prev = self.features[status == 1]
            good_next = next_pts[status == 1]
            
            if len(good_next) >= 3:
                displacements = good_next - good_prev
                
                # Exclude static background "anchor" points when target is moving
                mags = np.linalg.norm(displacements, axis=1)
                max_mag = np.max(mags)
                if max_mag > 1.0:
                    moving_mask = mags > 0.4
                    if np.sum(moving_mask) >= 3:
                        good_prev = good_prev[moving_mask]
                        good_next = good_next[moving_mask]
                        displacements = displacements[moving_mask]
                
                # Filter displacements to use only points closest to the bounding box center,
                # which helps avoid tracking static background elements.
                x1, y1, x2, y2 = self.box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                # Distance of each feature point to the box center
                dists = np.linalg.norm(good_prev - np.array([cx, cy]), axis=1)
                
                # Keep the closest features (e.g. 60%) to calculate shift
                num_to_keep = max(3, int(len(good_prev) * 0.6))
                closest_indices = np.argsort(dists)[:num_to_keep]
                
                relevant_displacements = displacements[closest_indices]
                dx = np.median(relevant_displacements[:, 0])
                dy = np.median(relevant_displacements[:, 1])
                
                # Shift box coordinates
                h, w = frame_gray.shape
                
                new_x1 = max(0, min(w - 10, int(x1 + dx)))
                new_y1 = max(0, min(h - 10, int(y1 + dy)))
                new_x2 = max(new_x1 + 10, min(w, int(x2 + dx)))
                new_y2 = max(new_y1 + 10, min(h, int(y2 + dy)))
                
                self.box = [new_x1, new_y1, new_x2, new_y2]
                self.prev_gray = frame_gray.copy()
                self.features = good_next.reshape(-1, 1, 2)
                
                return True, self.box
                
        self.active = False
        return False, None
