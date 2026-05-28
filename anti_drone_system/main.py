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
from control import PNGuidanceController, DirectPursuitController
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
    active_controller = "PN_GUIDANCE"  # Can toggle to "DIRECT_PURSUIT"

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
    logger.info("  [c] : Cycle Controller Law (PN Guidance <=> Direct Pursuit)")
    logger.info("  [d] : Toggle Debug Overlay (Projected vectors and predictions)")
    logger.info("  [a] : Send ARM Command to Drone")
    logger.info("  [s] : Send DISARM Command to Drone")
    logger.info("  [o] : Send TAKEOFF Command (Default 3m altitude)")
    logger.info("  [l] : Send LAND Command to Drone")
    logger.info("  [r] : Send Return-To-Launch (RTL) Command")
    logger.info("  [e] : EMERGENCY STOP (Hover immediately & disable auto-guidance)")
    logger.info("------------------------------------------")

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
                    cv2.imshow("Anti-Drone Intercept Feed", frame)
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
                # Lower threshold if target is already locked to allow recovery hysteresis (min 0.40 to prevent noise locks)
                active_conf = max(0.40, config['yolo']['confidence_threshold'] - 0.10) if lk_tracker.active else config['yolo']['confidence_threshold']
                detections = detector.detect(frame, conf_threshold=active_conf)
                if len(detections) > 0:
                    formatted_dets = [f"Box: {[int(coord) for coord in det[:4]]}, Conf: {det[4]:.2f}" for det in detections]
                    logger.info(f"YOLO Detections: {', '.join(formatted_dets)}")

            # 3. Update ByteTrack tracker
            active_tracks = tracker.update(detections)

            # 4. Associate Lucas-Kanade optical flow tracker with YOLO/ByteTrack detections
            best_iou = 0.0
            best_det = None
            if lk_tracker.active and lk_updated and len(detections) > 0:
                lx1, ly1, lx2, ly2 = lk_box
                w_l = lx2 - lx1
                h_l = ly2 - ly1
                cx_l = lx1 + w_l / 2.0
                cy_l = ly1 + h_l / 2.0
                
                for det in detections:
                    dx1, dy1, dx2, dy2 = det[:4]
                    w_d = dx2 - dx1
                    h_d = dy2 - dy1
                    cx_d = dx1 + w_d / 2.0
                    cy_d = dy1 + h_d / 2.0
                    
                    iou = calculate_iou([cx_d, cy_d, w_d, h_d], [cx_l, cy_l, w_l, h_l])
                    if iou > 0.20 and iou > best_iou:
                        best_iou = iou
                        best_det = det

            target_track = None
            from_yolo = False

            # A. Check if ByteTracker currently has the locked target
            if current_target_id is not None:
                for track in active_tracks:
                    if track.track_id == current_target_id:
                        target_track = track
                        from_yolo = True
                        break

            # B. If target was lost in ByteTrack but we have a matching YOLO detection, associate/acquire it
            # Force the new track to inherit the old current_target_id so the ID never changes during movement
            if current_target_id is not None and target_track is None and best_det is not None:
                for track in active_tracks:
                    tx1, ty1, tx2, ty2 = track.tlbr
                    iou = calculate_iou(track.xywh, [ (best_det[0]+best_det[2])/2.0, (best_det[1]+best_det[3])/2.0, best_det[2]-best_det[0], best_det[3]-best_det[1] ])
                    if iou > 0.40:  # More tolerant IoU matching for moving targets
                        track.track_id = current_target_id
                        target_track = track
                        from_yolo = True
                        break
                
                if target_track is None:
                    target_track = MockLKTrack(current_target_id, best_det[:4], score=best_det[4])
                    from_yolo = True

            # C. If YOLO missed but LK is active and updated, use pure LK tracking fallback
            if current_target_id is not None and target_track is None and lk_tracker.active and lk_updated:
                missing_yolo_frames += 1
                lk_max_frames = config['tracker'].get('lk_max_fallback_frames', 15)
                if missing_yolo_frames <= lk_max_frames:
                    lk_tracker.last_conf *= 0.95 # Decay confidence
                    target_track = MockLKTrack(current_target_id, lk_box, score=lk_tracker.last_conf)
                    from_yolo = False
                else:
                    logger.info(f"Target ID {current_target_id} lost (missing YOLO confirmation).")
                    lk_tracker.active = False
                    current_target_id = None
            elif current_target_id is not None and target_track is None:
                # Neither YOLO nor LK could track the drone
                logger.info(f"Target ID {current_target_id} lost.")
                current_target_id = None
                lk_tracker.active = False

            # D. If target_track is found, initialize/refresh LK tracker to lock coordinates
            if target_track is not None:
                if from_yolo:
                    x1, y1, x2, y2 = target_track.tlbr
                    lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                    lk_tracker.last_conf = target_track.score
                    missing_yolo_frames = 0
            else:
                # E. Target acquisition (if we don't have any locked target)
                if len(active_tracks) > 0:
                    active_tracks.sort(key=lambda x: x.score, reverse=True)
                    best_cand = active_tracks[0]
                    # Acquire target immediately on high score to eliminate delay
                    if best_cand.score >= config['tracker']['track_threshold']:
                        target_track = best_cand
                        current_target_id = target_track.track_id
                        x1, y1, x2, y2 = target_track.tlbr
                        lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                        lk_tracker.last_conf = target_track.score
                        missing_yolo_frames = 0
                        predictor.reset()
                        controller_pn.reset()
                        controller_pursuit.reset()
                        logger.info(f"Acquired target ID: {current_target_id}")

            # Initialize command variables
            cmd = None
            predicted_px = None

            # 5. If target is locked, run Ranging, Prediction, and Guidance
            if target_track is not None:
                cx, cy, w_box, h_box = target_track.xywh
                
                # Ranging
                distance = ranging.estimate(target_track.track_id, w_box, h_box)
                
                # Update 3D Motion Predictor
                img_h, img_w = frame.shape[:2]
                smoothed_pos, smoothed_vel = predictor.update(
                    target_track.track_id, cx, cy, distance, img_w, img_h
                )
                
                # Future location prediction (5 steps ahead to compensate camera-to-flight latency)
                pred_pos_3d = predictor.predict_future(steps=5)
                if pred_pos_3d is not None:
                    predicted_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                
                # 6. Compute Guidance Commands
                if active_controller == "PN_GUIDANCE":
                    cmd = controller_pn.compute_commands(
                        target_track, distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                    )
                else:
                    # Direct Pursuit (rule-based target following)
                    cmd = controller_pursuit.compute_commands(
                        target_track, distance, img_w, img_h
                    )
                
                # Send commands via MAVLink (if guidance switch and arming states are valid)
                if guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(
                        cmd['vx'], cmd['vy'], cmd['vz'], cmd['yaw_rate']
                    )
                
                # Render target overlay
                draw_target(frame, target_track, distance, predicted_px if debug_mode else None)
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
                cmd if debug_mode else None, 
                warnings_list
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
                cv2.imshow("Anti-Drone Intercept Feed", frame)
                
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
                elif key == ord('c'):
                    # Cycle Controller
                    active_controller = "DIRECT_PURSUIT" if active_controller == "PN_GUIDANCE" else "PN_GUIDANCE"
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
