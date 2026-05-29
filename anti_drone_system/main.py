import os
import sys

# Limit threads to prevent OpenBLAS memory allocation failures on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import cv2
import yaml
import logging
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import YOLODetector
from tracker import ByteTracker, BoundingBoxTracker
from ranging import DistanceEstimator
from prediction import MotionPredictor
from control import PNGuidanceController, DirectPursuitController, FollowTargetController
from mavlink import MAVLinkConnectionManager
from utils import ThreadedVideoStream, draw_hud, draw_target
from simulation import SITLSimulator

class MockLKTrack:
    def __init__(self, track_id, xyxy, score=1.0):
        self.track_id = track_id
        self._xyxy = xyxy
        self.score = score
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        self._xywh = [cx, cy, w, h]

    @property
    def xywh(self):
        return self._xywh

    @property
    def tlbr(self):
        return self._xyxy

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes in [x_c, y_c, w, h] format.
    """
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2
    
    x1_min, y1_min = xc1 - w1/2.0, yc1 - h1/2.0
    x1_max, y1_max = xc1 + w1/2.0, yc1 + h1/2.0
    x2_min, y2_min = xc2 - w2/2.0, yc2 - h2/2.0
    x2_max, y2_max = xc2 + w2/2.0, yc2 + h2/2.0
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

def compute_template_similarity(candidate_crop, template):
    """
    Computes visual similarity between candidate crop and target template using Normalized Cross-Correlation.
    """
    try:
        # Resize both to 64x64
        cand_resized = cv2.resize(candidate_crop, (64, 64))
        temp_resized = cv2.resize(template, (64, 64))
        
        # Grayscale conversion
        cand_gray = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.cvtColor(temp_resized, cv2.COLOR_BGR2GRAY)
        
        # Compute TM_CCOEFF_NORMED template matching
        res = cv2.matchTemplate(cand_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        # Color Histogram Correlation comparison
        cand_hsv = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2HSV)
        temp_hsv = cv2.cvtColor(temp_resized, cv2.COLOR_BGR2HSV)
        
        hist_cand = cv2.calcHist([cand_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist_temp = cv2.calcHist([temp_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        
        cv2.normalize(hist_cand, hist_cand, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_temp, hist_temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_score = cv2.compareHist(hist_cand, hist_temp, cv2.HISTCMP_CORREL)
        
        # Combine both scores: 70% structural similarity (NCC) + 30% color histogram similarity
        combined_score = 0.7 * max_val + 0.3 * hist_score
        return combined_score
    except Exception:
        return 0.0

def show_frame_fitted(window_name, img):
    """
    Displays the image in the specified window, automatically resizing it
    to match the current window dimensions to eliminate grey borders/margins.
    """
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and rect[2] > 0 and rect[3] > 0:
            img_disp = cv2.resize(img, (rect[2], rect[3]), interpolation=cv2.INTER_LINEAR)
        else:
            img_disp = img
    except Exception:
        img_disp = img
    cv2.imshow(window_name, img_disp)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AntiDroneSystem.Main")

def load_config(config_path):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to parse configuration file: {e}")
            sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Anti-Drone Vision & Guidance Pipeline for SpeedyBee/ArduPilot")
    parser.add_argument("--mode", type=str, choices=["simulation", "webcam", "sitl", "hardware"], help="System running mode")
    parser.add_argument("--model", type=str, help="Override YOLO model path")
    parser.add_argument("--source", type=str, help="Override camera index or video file source")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV GUI visualization window")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "settings.yaml")
    config = load_config(config_path)

    # Apply CLI overrides if provided
    if args.mode:
        config['system']['mode'] = args.mode
    if args.model:
        config['yolo']['model_path'] = args.model
    if args.source:
        config['camera']['source'] = args.source
    if args.no_gui:
        config['system']['gui'] = False

    logger.info("Initializing Anti-Drone Vision & Guidance Pipeline for SpeedyBee/ArduPilot...")

    # Parse and establish system mode
    mode = config['system'].get('mode', 'simulation').lower()
    logger.info(f"System operating mode: {mode.upper()}")

    # Map system mode to MAVLink mock simulation flag
    simulation_mode = (mode in ["simulation", "webcam"])
    config['system']['simulation_mode'] = simulation_mode

    # Initialize MAVLink Connection Manager
    mavlink_mgr = MAVLinkConnectionManager(config)
    mavlink_mgr.connect()

    # Initialize Detector
    detector = YOLODetector(config)

    # Initialize Tracker
    tracker = ByteTracker(config)

    # Initialize Range Estimator
    ranging = DistanceEstimator(config)

    # Initialize Motion Predictor
    predictor = MotionPredictor(config)

    # Initialize Guidance Controllers
    controller_pn = PNGuidanceController(config)
    controller_pursuit = DirectPursuitController(config)
    controller_follow = FollowTargetController(config)
    active_controller = "FOLLOW_TARGET"  # Default mode

    # Initialize Simulator target if virtual target mode is active
    virtual_target_mode = (mode == "simulation")
    sitl_sim = None
    if virtual_target_mode:
        sitl_sim = SITLSimulator(config)
        logger.info("SITL virtual target simulator activated. Injecting virtual moving target.")

    # Initialize Video Capture Stream
    video_stream = ThreadedVideoStream(config)
    video_stream.start()

    # Configure recording if requested
    save_output = config['system']['save_output']
    video_writer = None
    if save_output:
        output_dir = config['system']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"anti_drone_run_{timestamp}.mp4")
        logger.info(f"Video logging enabled. Output path: {output_path}")

    # Runtime states
    running = True
    paused = False
    guidance_active = False  # Safe switch: off by default, toggle with 't'
    FOLLOW_TARGET_ENABLED = False  # Follow safety switch: off by default, toggle with 'f'
    target_templates = []  # List of cropped images of the drone for visual ReID
    last_template_saved_time = 0.0
    lock_state = "ACQUISITION"  # Target lock states: ACQUISITION, LOCKED, LOST, REACQUISITION
    lost_time = 0.0
    last_w_box = 50.0
    last_h_box = 50.0
    last_known_box = None  # Store last known target box [cx, cy, w, h]
    debug_mode = config['system']['debug_overlay']
    current_target_id = None
    
    # Hybrid optical flow tracker initialization
    lk_tracker = BoundingBoxTracker()
    missing_yolo_frames = 0
    
    # Timing and FPS calculations
    last_frame_time = time.time()
    fps_accum = 0
    fps_frames = 0
    current_fps = 0.0

    # Command HUD bindings information printed
    logger.info("--- KEYBOARD CONTROLS IN OpenCV WINDOW ---")
    logger.info("  [q] : Quit Program")
    logger.info("  [p] : Pause Pipeline Execution")
    logger.info("  [t] : Toggle Guidance Switch (ENABLE/DISABLE Autonomous Pursuit)")
    logger.info("  [f] : Toggle Follow Mode Safety Switch (ENABLE/DISABLE active following)")
    logger.info("  [c] : Cycle Controller Law (Follow Target <=> PN Guidance <=> Direct Pursuit)")
    logger.info("  [d] : Toggle Debug Overlay (Projected vectors and predictions)")
    logger.info("  [a] : Send ARM Command to Drone")
    logger.info("  [s] : Send DISARM Command to Drone")
    logger.info("  [o] : Send TAKEOFF Command (Default 3m altitude)")
    logger.info("  [l] : Send LAND Command to Drone")
    logger.info("  [r] : Send Return-To-Launch (RTL) Command")
    logger.info("  [e] : EMERGENCY STOP (Hover immediately & disable auto-guidance)")
    logger.info("------------------------------------------")

    # Initialize OpenCV window once if GUI is enabled
    if config['system']['gui']:
        cv2.namedWindow("Anti-Drone Intercept Feed", cv2.WINDOW_NORMAL)

    try:
        while running:
            # 1. Capture/Read frame (with event wait to sync rate-limiting and avoid backlog queue latency)
            grabbed, frame = video_stream.read(wait=(not virtual_target_mode), timeout=0.1)
            if not grabbed and not virtual_target_mode:
                if video_stream.stopped:
                    logger.error("Video stream thread has stopped. Exiting pipeline...")
                    break
                logger.warning("No frame retrieved from camera. Retrying...")
                time.sleep(0.1)
                continue

            # In virtual target mode, if there's no camera, create a black canvas
            if frame is None and virtual_target_mode:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            if paused:
                if config['system']['gui']:
                    show_frame_fitted("Anti-Drone Intercept Feed", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                    elif key == ord('p'):
                        paused = False
                        logger.info("Pipeline Resumed.")
                continue

            # Compute dt for guidance
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            # Prepare grayscale frame for optical flow tracking
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Update Lucas-Kanade tracker first
            lk_updated = False
            lk_box = None
            if lk_tracker.active:
                lk_updated, lk_box = lk_tracker.update_tracker(frame_gray)

            detections = []

            # 2. Get detections (YOLO vs. Virtual Target Injection)
            if virtual_target_mode:
                # Local telemetry position and attitude from master link
                interceptor_pos = np.array([mavlink_mgr.local_x, mavlink_mgr.local_y, mavlink_mgr.local_z])
                interceptor_att = np.array([mavlink_mgr.roll, mavlink_mgr.pitch, mavlink_mgr.yaw])
                
                img_h, img_w = frame.shape[:2]
                sim_detection, rel_3d = sitl_sim.project_target_to_camera(
                    interceptor_pos, interceptor_att, img_w, img_h
                )
                
                # Draw the virtual drone onto the frame
                frame = sitl_sim.generate_virtual_frame(sim_detection, img_w, img_h)
                
                # Bypass YOLO and use projected coordinate directly for tracker testing
                if sim_detection is not None:
                    detections = [sim_detection]
            else:
                # Hardware mode: Run YOLO Detection
                # Lower threshold if target is already locked to allow recovery hysteresis (min 0.20 to prevent noise locks)
                active_conf = max(0.20, config['yolo']['confidence_threshold'] - 0.10) if lk_tracker.active else config['yolo']['confidence_threshold']
                detections = detector.detect(frame, conf_threshold=active_conf)
                if len(detections) > 0:
                    formatted_dets = [f"Box: {[int(coord) for coord in det[:4]]}, Conf: {det[4]:.2f}" for det in detections]
                    logger.info(f"YOLO Detections: {', '.join(formatted_dets)}")

            # 3. Update ByteTrack tracker
            active_tracks = tracker.update(detections)

            target_track = None
            from_yolo = False

            # Get predicted/last known target box coordinates for association matching
            pred_cx, pred_cy, pred_w, pred_h = None, None, None, None
            if current_target_id is not None:
                pred_pos_3d = predictor.state[:3]
                img_h, img_w = frame.shape[:2]
                pred_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                if pred_px is not None and predictor.covariance[0,0] < 5000:
                    pred_cx, pred_cy = pred_px
                    pred_w = (ranging.focal_length * ranging.drone_real_size) / max(0.1, pred_pos_3d[2])
                    pred_h = pred_w
                elif last_known_box is not None:
                    pred_cx, pred_cy, pred_w, pred_h = last_known_box

            # Find if any active track or YOLO detection matches the target (current_target_id)
            matched_track = None
            matched_det = None
            best_assoc_score = 0.0

            if current_target_id is not None:
                # 1. Match with active tracks first
                for track in active_tracks:
                    txc, tyc, tw, th = track.xywh
                    
                    iou_pred = 0.0
                    if pred_cx is not None:
                        iou_pred = calculate_iou([txc, tyc, tw, th], [pred_cx, pred_cy, pred_w, pred_h])
                    
                    iou_lk = 0.0
                    if lk_tracker.active and lk_updated and lk_box is not None:
                        lx1, ly1, lx2, ly2 = lk_box
                        lcx = lx1 + (lx2 - lx1)/2.0
                        lcy = ly1 + (ly2 - ly1)/2.0
                        lw = lx2 - lx1
                        lh = ly2 - ly1
                        iou_lk = calculate_iou([txc, tyc, tw, th], [lcx, lcy, lw, lh])
                        
                    dist_score = 0.0
                    if pred_cx is not None:
                        dist = np.hypot(txc - pred_cx, tyc - pred_cy)
                        max_dist = max(300.0, frame.shape[1] * 0.5)
                        if dist < max_dist:
                            dist_score = 1.0 - (dist / max_dist)
                            
                    # Associate based on highest overlap or proximity
                    assoc_score = max(iou_pred, iou_lk, dist_score * 0.5)
                    if assoc_score > 0.20 and assoc_score > best_assoc_score:
                        best_assoc_score = assoc_score
                        matched_track = track
                        matched_det = None
                        
                # 2. Match with raw YOLO detections if no active track matched
                if matched_track is None:
                    for det in detections:
                        dx1, dy1, dx2, dy2, conf = det[:5]
                        dxc = (dx1 + dx2) / 2.0
                        dyc = (dy1 + dy2) / 2.0
                        dw = dx2 - dx1
                        dh = dy2 - dy1
                        
                        iou_pred = 0.0
                        if pred_cx is not None:
                            iou_pred = calculate_iou([dxc, dyc, dw, dh], [pred_cx, pred_cy, pred_w, pred_h])
                            
                        iou_lk = 0.0
                        if lk_tracker.active and lk_updated and lk_box is not None:
                            lx1, ly1, lx2, ly2 = lk_box
                            lcx = lx1 + (lx2 - lx1)/2.0
                            lcy = ly1 + (ly2 - ly1)/2.0
                            lw = lx2 - lx1
                            lh = ly2 - ly1
                            iou_lk = calculate_iou([dxc, dyc, dw, dh], [lcx, lcy, lw, lh])
                            
                        dist_score = 0.0
                        if pred_cx is not None:
                            dist = np.hypot(dxc - pred_cx, dyc - pred_cy)
                            max_dist = max(300.0, frame.shape[1] * 0.5)
                            if dist < max_dist:
                                dist_score = 1.0 - (dist / max_dist)
                                
                        assoc_score = max(iou_pred, iou_lk, dist_score * 0.5)
                        if assoc_score > 0.20 and assoc_score > best_assoc_score:
                            best_assoc_score = assoc_score
                            matched_det = det
                            matched_track = None

            # Visual Template Re-Identification (ReID) Match
            reid_matched_track = None
            reid_matched_det = None
            best_sim_score = 0.0
            
            if len(target_templates) > 0:
                # Compare active tracks to templates
                for track in active_tracks:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop = frame[y1:y2, x1:x2]
                        for temp in target_templates:
                            sim = compute_template_similarity(crop, temp)
                            if sim > best_sim_score:
                                best_sim_score = sim
                                reid_matched_track = track
                                reid_matched_det = None
                                
                # Compare raw detections to templates
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop = frame[y1:y2, x1:x2]
                        for temp in target_templates:
                            sim = compute_template_similarity(crop, temp)
                            if sim > best_sim_score:
                                best_sim_score = sim
                                reid_matched_track = None
                                reid_matched_det = det
            
            # If we have a very strong visual template match, override standard distance matches
            if best_sim_score >= 0.70:
                if reid_matched_track is not None:
                    if current_target_id is None:
                        current_target_id = reid_matched_track.track_id
                    reid_matched_track.track_id = current_target_id
                    matched_track = reid_matched_track
                    logger.info(f"Visual ReID: Matched target ID {current_target_id} with score {best_sim_score:.2f}")
                elif reid_matched_det is not None:
                    if current_target_id is None:
                        current_target_id = 99 # Default custom ReID ID for raw detection
                    matched_det = reid_matched_det
                    logger.info(f"Visual ReID: Matched raw YOLO detection as target ID {current_target_id} with score {best_sim_score:.2f}")

            # Fallback: if no strong match but there is only one active track or YOLO detection in the frame,
            # and we are currently tracking a lost target, associate it to maintain lock continuity.
            if current_target_id is not None and matched_track is None and matched_det is None:
                if len(active_tracks) == 1:
                    matched_track = active_tracks[0]
                    logger.info(f"Lock continuity: matching single active track to target ID {current_target_id}")
                elif len(detections) == 1:
                    matched_det = detections[0]
                    logger.info(f"Lock continuity: matching single YOLO detection to target ID {current_target_id}")

            # Apply association results
            if matched_track is not None:
                matched_track.track_id = current_target_id
                target_track = matched_track
                from_yolo = True
            elif matched_det is not None:
                target_track = MockLKTrack(current_target_id, matched_det[:4], score=matched_det[4])
                from_yolo = True
            elif current_target_id is not None and lk_tracker.active and lk_updated and lk_box is not None:
                # Fallback to optical flow tracking if YOLO missed
                missing_yolo_frames += 1
                lk_max_frames = config['tracker'].get('lk_max_fallback_frames', 15)
                if missing_yolo_frames <= lk_max_frames:
                    lk_tracker.last_conf *= 0.95
                    target_track = MockLKTrack(current_target_id, lk_box, score=lk_tracker.last_conf)
                    from_yolo = False
                else:
                    logger.info(f"Target ID {current_target_id} lost (missing YOLO confirmation timeout).")
                    lk_tracker.active = False
                    target_track = None
            else:
                if current_target_id is not None:
                    lk_tracker.active = False

            # Aggressive Reacquisition: If target is LOST and we couldn't associate it with the old ID,
            # but there is a fresh high-confidence track/detection in the frame, re-lock onto it immediately
            # under the original target ID instead of waiting for the 5-second timeout.
            if target_track is None and current_target_id is not None and lock_state == "LOST":
                best_cand = None
                if len(active_tracks) > 0:
                    active_tracks.sort(key=lambda x: x.score, reverse=True)
                    if active_tracks[0].score >= config['tracker']['track_threshold']:
                        best_cand = active_tracks[0]
                        best_cand.track_id = current_target_id
                        target_track = best_cand
                        from_yolo = True
                        logger.info(f"Aggressive reacquisition of target ID {current_target_id} from active track.")
                
                if best_cand is None and len(detections) > 0:
                    detections_sort = sorted(detections, key=lambda x: x[4], reverse=True)
                    if detections_sort[0][4] >= config['tracker']['track_threshold']:
                        best_det = detections_sort[0]
                        target_track = MockLKTrack(current_target_id, best_det[:4], score=best_det[4])
                        from_yolo = True
                        logger.info(f"Aggressive reacquisition of target ID {current_target_id} from raw YOLO detection.")

            # State Machine Transitions based on target tracking status
            if target_track is not None:
                if lock_state == "ACQUISITION":
                    lock_state = "LOCKED"
                    logger.info(f"Acquired target ID: {current_target_id}")
                elif lock_state == "LOST":
                    lock_state = "REACQUISITION"
                    logger.info(f"Target ID {current_target_id} reacquired!")
                elif lock_state == "REACQUISITION":
                    lock_state = "LOCKED"
                
                # Re-initialize LK tracker features from YOLO detection to prevent optical flow drift
                if from_yolo:
                    x1, y1, x2, y2 = target_track.tlbr
                    lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                    lk_tracker.last_conf = target_track.score
                    missing_yolo_frames = 0
            else:
                if current_target_id is not None:
                    if lock_state in ["LOCKED", "REACQUISITION"]:
                        lock_state = "LOST"
                        lost_time = time.time()
                        logger.info(f"Target ID {current_target_id} direct tracking lost. Entering prediction search.")
                    
                    # Manage timeouts in LOST state
                    elapsed_lost = time.time() - lost_time
                    reacq_timeout = config['guidance'].get('reacquisition_timeout', 5.0)
                    loss_timeout = config['guidance'].get('target_loss_timeout', 3.0)
                    
                    if elapsed_lost > reacq_timeout:
                        logger.info(f"Target ID {current_target_id} completely lost after timeout.")
                        lock_state = "ACQUISITION"
                        current_target_id = None
                        last_known_box = None
                        predictor.reset()
                        controller_pn.reset()
                        controller_pursuit.reset()
                        controller_follow.reset()
                        
                        if config['guidance'].get('rtl_after_timeout', False) and mavlink_mgr.is_connected:
                            logger.warning("Failsafe: Switching vehicle to RTL due to target loss.")
                            mavlink_mgr.set_mode("RTL")
                    elif elapsed_lost > loss_timeout:
                        if guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                            mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                    else:
                        # Extrapolate position using dead reckoning (Kalman state transition)
                        predictor.state = np.dot(predictor.F, predictor.state)
                        predictor.covariance = np.dot(predictor.F, np.dot(predictor.covariance, predictor.F.T)) + predictor.Q
                        
                        smoothed_pos = predictor.state[:3]
                        smoothed_vel = predictor.state[3:]
                        
                        img_h, img_w = frame.shape[:2]
                        pred_px = predictor.project_to_image(smoothed_pos, img_w, img_h)
                        if pred_px is not None:
                            cx_p, cy_p = pred_px
                            box_sz = (ranging.focal_length * ranging.drone_real_size) / max(0.1, smoothed_pos[2])
                            x1_p = cx_p - box_sz / 2.0
                            y1_p = cy_p - box_sz / 2.0
                            x2_p = cx_p + box_sz / 2.0
                            y2_p = cy_p + box_sz / 2.0
                            
                            target_track = MockLKTrack(current_target_id, [x1_p, y1_p, x2_p, y2_p], score=0.4)
                            distance = smoothed_pos[2]
                else:
                    if len(active_tracks) > 0:
                        active_tracks.sort(key=lambda x: x.score, reverse=True)
                        best_cand = active_tracks[0]
                        if best_cand.score >= config['tracker']['track_threshold']:
                            target_track = best_cand
                            current_target_id = target_track.track_id
                            lock_state = "LOCKED"
                            x1, y1, x2, y2 = target_track.tlbr
                            lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                            lk_tracker.last_conf = target_track.score
                            missing_yolo_frames = 0
                            predictor.reset()
                            controller_pn.reset()
                            controller_pursuit.reset()
                            controller_follow.reset()
                            logger.info(f"Acquired target ID: {current_target_id}")

            # Initialize command variables
            cmd = None
            predicted_px = None

            # 5. If target is locked, run Ranging, Prediction, and Guidance
            if target_track is not None:
                cx, cy, w_box, h_box = target_track.xywh
                last_w_box, last_h_box = w_box, h_box
                last_known_box = [cx, cy, w_box, h_box]
                
                # Save visual templates if target is locked and follow mode is active
                if FOLLOW_TARGET_ENABLED and lock_state in ["LOCKED", "REACQUISITION"] and from_yolo:
                    x1_t, y1_t, x2_t, y2_t = map(int, target_track.tlbr)
                    x1_t = max(0, x1_t)
                    y1_t = max(0, y1_t)
                    x2_t = min(frame.shape[1], x2_t)
                    y2_t = min(frame.shape[0], y2_t)
                    
                    if (x2_t - x1_t) > 10 and (y2_t - y1_t) > 10:
                        # Throttled template saving to capture different angles and scale variations
                        if len(target_templates) < 10:
                            if (time.time() - last_template_saved_time) > 0.5:
                                crop_img = frame[y1_t:y2_t, x1_t:x2_t].copy()
                                target_templates.append(crop_img)
                                last_template_saved_time = time.time()
                                logger.info(f"Target visual template saved to memory. Size: {len(target_templates)}")
                
                img_h, img_w = frame.shape[:2]
                if lock_state in ["LOCKED", "REACQUISITION"]:
                    distance = ranging.estimate(target_track.track_id, w_box, h_box)
                    smoothed_pos, smoothed_vel = predictor.update(
                        target_track.track_id, cx, cy, distance, img_w, img_h
                    )
                else:
                    # Predicted state is already updated in Kalman predict step above
                    pass
                
                # Future location prediction (5 steps ahead to compensate camera-to-flight latency)
                pred_pos_3d = predictor.predict_future(steps=5)
                if pred_pos_3d is not None:
                    predicted_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                
                # 6. Compute Guidance Commands
                if active_controller == "PN_GUIDANCE":
                    cmd = controller_pn.compute_commands(
                        target_track, distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                    )
                elif active_controller == "FOLLOW_TARGET":
                    cmd = controller_follow.compute_commands(
                        target_track, distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                    )
                else:
                    # Direct Pursuit (rule-based target following)
                    cmd = controller_pursuit.compute_commands(
                        target_track, distance, img_w, img_h
                    )
                
                # Send commands via MAVLink (if guidance switch, follow switch, and arming states are valid)
                if FOLLOW_TARGET_ENABLED and guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(
                        cmd['vx'], cmd['vy'], cmd['vz'], cmd['yaw_rate']
                    )
                elif guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    # Safety Switch Off (Observation Mode) -> Hover
                    mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                
                # Render target overlay
                draw_target(frame, target_track, distance, predicted_px if debug_mode else None)
                
                # Draw prediction label if target is lost
                if lock_state == "LOST":
                    x1, y1, _, _ = map(int, target_track.tlbr)
                    cv2.putText(frame, "DEAD RECKONING ACTIVE", (x1, y1 - 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                # 7. Safety Target-Loss Failsafe:
                # If target is lost and guidance is active, command hover (stop velocity)
                if guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)

            # Clean tracker history
            removed_ids = [t.track_id for t in tracker.removed_stracks]
            for r_id in removed_ids:
                ranging.clean_track(r_id)
            tracker.removed_stracks.clear()

            # 7. Collect Safety Alerts & Warnings
            warnings_list = []
            
            # Telemetry Connection Check
            if not mavlink_mgr.is_connected and not simulation_mode:
                warnings_list.append("WARNING: MAVLINK DISCONNECTED")
                
            # Mode Check
            if guidance_active and mavlink_mgr.is_connected and mavlink_mgr.current_mode != "GUIDED":
                warnings_list.append("WARNING: VEHICLE NOT IN GUIDED MODE")
                
            # Low Loop FPS Check
            if current_fps > 0 and current_fps < config['system']['min_fps_warning']:
                warnings_list.append(f"WARNING: LOW LOOP FPS ({current_fps:.1f})")
                
            # Target Confidence warning
            if target_track is not None and target_track.score < config['system']['low_confidence_warning']:
                warnings_list.append(f"WARNING: LOW TARGET CONFIDENCE ({target_track.score:.2f})")

            # Camera feed black frame check
            if frame is not None and not virtual_target_mode:
                if np.mean(frame) < 2.0:
                    warnings_list.append("WARNING: BLACK IMAGE - CHECK CAM INDEX (--source)")

            # 8. Render HUD HUD overlay
            draw_hud(
                frame, 
                (target_track is not None), 
                current_fps, 
                mavlink_mgr, 
                active_controller, 
                cmd, 
                warnings_list,
                target_id=current_target_id,
                lock_status=lock_state,
                follow_target_enabled=FOLLOW_TARGET_ENABLED,
                desired_distance=config['guidance'].get('desired_follow_distance', 5.0)
            )

            # Record frame if requested
            if save_output:
                if video_writer is None:
                    img_h, img_w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (img_w, img_h))
                video_writer.write(frame)

            # Show GUI Window
            if config['system']['gui']:
                show_frame_fitted("Anti-Drone Intercept Feed", frame)
                
                # Parse Key Binds
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting system...")
                    running = False
                elif key == ord('p'):
                    paused = True
                    logger.info("Pipeline Paused.")
                elif key == ord('t'):
                    guidance_active = not guidance_active
                    logger.info(f"Autonomous Guidance Switch: {'ENABLED' if guidance_active else 'DISABLED'}")
                elif key == ord('f'):
                    FOLLOW_TARGET_ENABLED = not FOLLOW_TARGET_ENABLED
                    logger.info(f"Follow Target Safety Switch: {'ENABLED (ACTIVE)' if FOLLOW_TARGET_ENABLED else 'DISABLED (OBSERVATION)'}")
                    if not FOLLOW_TARGET_ENABLED:
                        target_templates.clear()
                        logger.info("Cleared target visual templates from memory.")
                elif key == ord('c'):
                    # Cycle Controller between FOLLOW_TARGET, PN_GUIDANCE, DIRECT_PURSUIT
                    if active_controller == "FOLLOW_TARGET":
                        active_controller = "PN_GUIDANCE"
                    elif active_controller == "PN_GUIDANCE":
                        active_controller = "DIRECT_PURSUIT"
                    else:
                        active_controller = "FOLLOW_TARGET"
                    logger.info(f"Switched controller law to: {active_controller}")
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    logger.info(f"Debug Overlay: {'ENABLED' if debug_mode else 'DISABLED'}")
                elif key == ord('a'):
                    mavlink_mgr.arm()
                elif key == ord('s'):
                    mavlink_mgr.disarm()
                elif key == ord('o'):
                    mavlink_mgr.takeoff(altitude_m=3.0)
                elif key == ord('l'):
                    mavlink_mgr.set_mode("LAND")
                elif key == ord('r'):
                    mavlink_mgr.set_mode("RTL")
                elif key == ord('e'):
                    # EMERGENCY STOP
                    guidance_active = False
                    logger.warning("EMERGENCY STOP TRIGGERED: Guidance Switch DISABLED.")
                    if mavlink_mgr.is_connected:
                        mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
            else:
                time.sleep(0.005)

            # Loop Rate calculation
            fps_frames += 1
            fps_accum += dt
            if fps_accum >= 1.0:
                current_fps = fps_frames / fps_accum
                fps_frames = 0
                fps_accum = 0.0

    except KeyboardInterrupt:
        logger.info("System interrupted. Shutting down...")
    finally:
        logger.info("Cleaning up resources...")
        video_stream.stop()
        
        if video_writer:
            video_writer.release()
            
        if mavlink_mgr.is_connected:
            # Force zero velocity hover on shutdown
            if mavlink_mgr.is_armed and not mavlink_mgr.simulation_mode:
                logger.info("Sending safe hover command to SpeedyBee FC...")
                mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
            mavlink_mgr.disconnect()
            
        if config['system']['gui']:
            cv2.destroyAllWindows()
            
        logger.info("Anti-Drone Vision & Guidance Pipeline closed.")

if __name__ == "__main__":
    main()
