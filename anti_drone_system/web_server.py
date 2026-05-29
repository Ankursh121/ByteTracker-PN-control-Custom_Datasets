import os
import sys
import time
import cv2
import yaml
import logging
import asyncio
import threading
import subprocess
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Adjust Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import YOLODetector
from tracker import ByteTracker, BoundingBoxTracker
from ranging import DistanceEstimator
from prediction import MotionPredictor
from control import PNGuidanceController, DirectPursuitController, FollowTargetController
from mavlink import MAVLinkConnectionManager
from utils import ThreadedVideoStream, draw_hud, draw_target
from simulation import SITLSimulator

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AntiDroneSystem.WebAPI")

app = FastAPI(title="Anti-Drone Autonomous Pursuit Control Station")

# Allow CORS for React Frontend running on other port (e.g. 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to calculate box IoU and templates (copied from main.py)
def calculate_iou(box1, box2):
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
    try:
        cand_resized = cv2.resize(candidate_crop, (64, 64))
        temp_resized = cv2.resize(template, (64, 64))
        cand_gray = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.cvtColor(temp_resized, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(cand_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        cand_hsv = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2HSV)
        temp_hsv = cv2.cvtColor(temp_resized, cv2.COLOR_BGR2HSV)
        hist_cand = cv2.calcHist([cand_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist_temp = cv2.calcHist([temp_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist_cand, hist_cand, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_temp, hist_temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_score = cv2.compareHist(hist_cand, hist_temp, cv2.HISTCMP_CORREL)
        return 0.7 * max_val + 0.3 * hist_score
    except Exception:
        return 0.0

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


class PipelineManager:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "settings.yaml")
        self.config = {}
        self.load_config()
        
        self.running = False
        self.paused = False
        self.guidance_active = False
        self.follow_target_enabled = False
        self.active_controller = "FOLLOW_TARGET"
        self.debug_mode = True
        
        self.lock_state = "ACQUISITION"
        self.current_fps = 0.0
        self.current_target_id = None
        self.target_distance = 0.0
        self.warnings_list = []
        self.latest_frame_encoded = None
        self.last_known_box = None
        
        self.thread = None
        self.lock = threading.Lock()
        
        # System status data
        self.telemetry = {}
        self.current_cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        return self.config

    def save_config(self, new_config: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(new_config, f, default_flow_style=False)
        self.config = new_config

    def start_pipeline(self):
        with self.lock:
            if self.running:
                logger.info("Pipeline already running.")
                return False
            self.running = True
            self.thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self.thread.start()
            logger.info("Tracking pipeline thread started.")
            return True

    def stop_pipeline(self):
        with self.lock:
            if not self.running:
                return False
            self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)
            logger.info("Tracking pipeline thread stopped.")
        return True

    def _pipeline_loop(self):
        """ Runs the actual tracking / detection loop, streaming results back """
        config = self.load_config()
        
        # Override GUI to false since we serve over web, unless they want both
        config['system']['gui'] = False
        
        # Parse and establish system mode
        mode = config['system'].get('mode', 'simulation').lower()
        simulation_mode = (mode in ["simulation", "webcam"])
        config['system']['simulation_mode'] = simulation_mode

        # Initialize MAVLink Connection
        mavlink_mgr = MAVLinkConnectionManager(config)
        mavlink_mgr.connect()

        # Initialize core elements
        detector = YOLODetector(config)
        tracker = ByteTracker(config)
        ranging = DistanceEstimator(config)
        predictor = MotionPredictor(config)
        
        controller_pn = PNGuidanceController(config)
        controller_pursuit = DirectPursuitController(config)
        controller_follow = FollowTargetController(config)
        
        virtual_target_mode = (mode == "simulation")
        sitl_sim = None
        if virtual_target_mode:
            sitl_sim = SITLSimulator(config)

        # Video Capture
        video_stream = ThreadedVideoStream(config)
        video_stream.start()

        # Local variables
        target_templates = []
        last_template_saved_time = 0.0
        self.lock_state = "ACQUISITION"
        lost_time = 0.0
        self.last_known_box = None
        self.current_target_id = None
        self.target_distance = 0.0
        
        lk_tracker = BoundingBoxTracker()
        missing_yolo_frames = 0
        
        last_frame_time = time.time()
        fps_accum = 0
        fps_frames = 0

        logger.info("Pipeline processing loop active.")

        try:
            while self.running:
                # Loop rate limiting
                now = time.time()
                dt = now - last_frame_time
                last_frame_time = now
                
                # Check paused
                if self.paused:
                    time.sleep(0.03)
                    continue

                grabbed, frame = video_stream.read(wait=(not virtual_target_mode), timeout=0.1)
                if not grabbed and not virtual_target_mode:
                    if video_stream.stopped:
                        logger.error("Video stream stopped.")
                        break
                    time.sleep(0.01)
                    continue

                if frame is None and virtual_target_mode:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Update Lucas-Kanade tracker
                lk_updated = False
                lk_box = None
                if lk_tracker.active:
                    lk_updated, lk_box = lk_tracker.update_tracker(frame_gray)

                detections = []
                if virtual_target_mode:
                    interceptor_pos = np.array([mavlink_mgr.local_x, mavlink_mgr.local_y, mavlink_mgr.local_z])
                    interceptor_att = np.array([mavlink_mgr.roll, mavlink_mgr.pitch, mavlink_mgr.yaw])
                    img_h, img_w = frame.shape[:2]
                    sim_detection, rel_3d = sitl_sim.project_target_to_camera(
                        interceptor_pos, interceptor_att, img_w, img_h
                    )
                    frame = sitl_sim.generate_virtual_frame(sim_detection, img_w, img_h)
                    if sim_detection is not None:
                        detections = [sim_detection]
                else:
                    active_conf = max(0.20, config['yolo']['confidence_threshold'] - 0.10) if lk_tracker.active else config['yolo']['confidence_threshold']
                    detections = detector.detect(frame, conf_threshold=active_conf)

                active_tracks = tracker.update(detections)
                target_track = None
                from_yolo = False

                pred_cx, pred_cy, pred_w, pred_h = None, None, None, None
                if self.current_target_id is not None:
                    pred_pos_3d = predictor.state[:3]
                    img_h, img_w = frame.shape[:2]
                    pred_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                    if pred_px is not None and predictor.covariance[0,0] < 5000:
                        pred_cx, pred_cy = pred_px
                        pred_w = (ranging.focal_length * ranging.drone_real_size) / max(0.1, pred_pos_3d[2])
                        pred_h = pred_w
                    elif self.last_known_box is not None:
                        pred_cx, pred_cy, pred_w, pred_h = self.last_known_box

                matched_track = None
                matched_det = None
                best_assoc_score = 0.0

                if self.current_target_id is not None:
                    for track in active_tracks:
                        txc, tyc, tw, th = track.xywh
                        iou_pred = calculate_iou([txc, tyc, tw, th], [pred_cx, pred_cy, pred_w, pred_h]) if pred_cx is not None else 0.0
                        iou_lk = calculate_iou([txc, tyc, tw, th], [lk_box[0] + (lk_box[2]-lk_box[0])/2, lk_box[1] + (lk_box[3]-lk_box[1])/2, lk_box[2]-lk_box[0], lk_box[3]-lk_box[1]]) if (lk_tracker.active and lk_updated and lk_box is not None) else 0.0
                        dist_score = 0.0
                        if pred_cx is not None:
                            dist = np.hypot(txc - pred_cx, tyc - pred_cy)
                            max_dist = max(300.0, frame.shape[1] * 0.5)
                            if dist < max_dist:
                                dist_score = 1.0 - (dist / max_dist)
                        assoc_score = max(iou_pred, iou_lk, dist_score * 0.5)
                        if assoc_score > 0.20 and assoc_score > best_assoc_score:
                            best_assoc_score = assoc_score
                            matched_track = track
                            matched_det = None
                            
                    if matched_track is None:
                        for det in detections:
                            dx1, dy1, dx2, dy2, conf = det[:5]
                            dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                            dw, dh = dx2 - dx1, dy2 - dy1
                            iou_pred = calculate_iou([dxc, dyc, dw, dh], [pred_cx, pred_cy, pred_w, pred_h]) if pred_cx is not None else 0.0
                            iou_lk = calculate_iou([dxc, dyc, dw, dh], [lk_box[0] + (lk_box[2]-lk_box[0])/2, lk_box[1] + (lk_box[3]-lk_box[1])/2, lk_box[2]-lk_box[0], lk_box[3]-lk_box[1]]) if (lk_tracker.active and lk_updated and lk_box is not None) else 0.0
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

                reid_matched_track = None
                reid_matched_det = None
                best_sim_score = 0.0
                if len(target_templates) > 0:
                    for track in active_tracks:
                        x1, y1, x2, y2 = map(int, track.tlbr)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            crop = frame[y1:y2, x1:x2]
                            for temp in target_templates:
                                sim = compute_template_similarity(crop, temp)
                                if sim > best_sim_score:
                                    best_sim_score = sim
                                    reid_matched_track = track
                                    reid_matched_det = None
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det[:4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            crop = frame[y1:y2, x1:x2]
                            for temp in target_templates:
                                sim = compute_template_similarity(crop, temp)
                                if sim > best_sim_score:
                                    best_sim_score = sim
                                    reid_matched_track = None
                                    reid_matched_det = det

                if best_sim_score >= 0.70:
                    if reid_matched_track is not None:
                        if self.current_target_id is None:
                            self.current_target_id = reid_matched_track.track_id
                        reid_matched_track.track_id = self.current_target_id
                        matched_track = reid_matched_track
                    elif reid_matched_det is not None:
                        if self.current_target_id is None:
                            self.current_target_id = 99
                        matched_det = reid_matched_det

                if self.current_target_id is not None and matched_track is None and matched_det is None:
                    if len(active_tracks) == 1:
                        matched_track = active_tracks[0]
                    elif len(detections) == 1:
                        matched_det = detections[0]

                if matched_track is not None:
                    matched_track.track_id = self.current_target_id
                    target_track = matched_track
                    from_yolo = True
                elif matched_det is not None:
                    target_track = MockLKTrack(self.current_target_id, matched_det[:4], score=matched_det[4])
                    from_yolo = True
                elif self.current_target_id is not None and lk_tracker.active and lk_updated and lk_box is not None:
                    missing_yolo_frames += 1
                    lk_max_frames = config['tracker'].get('lk_max_fallback_frames', 15)
                    if missing_yolo_frames <= lk_max_frames:
                        lk_tracker.last_conf *= 0.95
                        target_track = MockLKTrack(self.current_target_id, lk_box, score=lk_tracker.last_conf)
                        from_yolo = False
                    else:
                        lk_tracker.active = False
                        target_track = None
                else:
                    if self.current_target_id is not None:
                        lk_tracker.active = False

                if target_track is None and self.current_target_id is not None and self.lock_state == "LOST":
                    best_cand = None
                    if len(active_tracks) > 0:
                        active_tracks.sort(key=lambda x: x.score, reverse=True)
                        if active_tracks[0].score >= config['tracker']['track_threshold']:
                            best_cand = active_tracks[0]
                            best_cand.track_id = self.current_target_id
                            target_track = best_cand
                            from_yolo = True
                    if best_cand is None and len(detections) > 0:
                        detections_sort = sorted(detections, key=lambda x: x[4], reverse=True)
                        if detections_sort[0][4] >= config['tracker']['track_threshold']:
                            best_det = detections_sort[0]
                            target_track = MockLKTrack(self.current_target_id, best_det[:4], score=best_det[4])
                            from_yolo = True

                if target_track is not None:
                    if self.lock_state == "ACQUISITION":
                        self.lock_state = "LOCKED"
                    elif self.lock_state == "LOST":
                        self.lock_state = "REACQUISITION"
                    elif self.lock_state == "REACQUISITION":
                        self.lock_state = "LOCKED"
                    
                    if from_yolo:
                        x1, y1, x2, y2 = target_track.tlbr
                        lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                        lk_tracker.last_conf = target_track.score
                        missing_yolo_frames = 0
                else:
                    if self.current_target_id is not None:
                        if self.lock_state in ["LOCKED", "REACQUISITION"]:
                            self.lock_state = "LOST"
                            lost_time = time.time()
                        
                        elapsed_lost = time.time() - lost_time
                        reacq_timeout = config['guidance'].get('reacquisition_timeout', 5.0)
                        loss_timeout = config['guidance'].get('target_loss_timeout', 3.0)
                        
                        if elapsed_lost > reacq_timeout:
                            self.lock_state = "ACQUISITION"
                            self.current_target_id = None
                            self.last_known_box = None
                            predictor.reset()
                            controller_pn.reset()
                            controller_pursuit.reset()
                            controller_follow.reset()
                            if config['guidance'].get('rtl_after_timeout', False) and mavlink_mgr.is_connected:
                                mavlink_mgr.set_mode("RTL")
                        elif elapsed_lost > loss_timeout:
                            if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                                mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                        else:
                            predictor.state = np.dot(predictor.F, predictor.state)
                            predictor.covariance = np.dot(predictor.F, np.dot(predictor.covariance, predictor.F.T)) + predictor.Q
                            smoothed_pos = predictor.state[:3]
                            img_h, img_w = frame.shape[:2]
                            pred_px = predictor.project_to_image(smoothed_pos, img_w, img_h)
                            if pred_px is not None:
                                cx_p, cy_p = pred_px
                                box_sz = (ranging.focal_length * ranging.drone_real_size) / max(0.1, smoothed_pos[2])
                                x1_p, y1_p = cx_p - box_sz / 2.0, cy_p - box_sz / 2.0
                                x2_p, y2_p = cx_p + box_sz / 2.0, cy_p + box_sz / 2.0
                                target_track = MockLKTrack(self.current_target_id, [x1_p, y1_p, x2_p, y2_p], score=0.4)
                                self.target_distance = smoothed_pos[2]
                    else:
                        if len(active_tracks) > 0:
                            active_tracks.sort(key=lambda x: x.score, reverse=True)
                            best_cand = active_tracks[0]
                            if best_cand.score >= config['tracker']['track_threshold']:
                                target_track = best_cand
                                self.current_target_id = target_track.track_id
                                self.lock_state = "LOCKED"
                                x1, y1, x2, y2 = target_track.tlbr
                                lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                                lk_tracker.last_conf = target_track.score
                                missing_yolo_frames = 0
                                predictor.reset()
                                controller_pn.reset()
                                controller_pursuit.reset()
                                controller_follow.reset()

                cmd = None
                predicted_px = None

                if target_track is not None:
                    cx, cy, w_box, h_box = target_track.xywh
                    self.last_known_box = [cx, cy, w_box, h_box]
                    
                    if self.follow_target_enabled and self.lock_state in ["LOCKED", "REACQUISITION"] and from_yolo:
                        x1_t, y1_t, x2_t, y2_t = map(int, target_track.tlbr)
                        x1_t, y1_t = max(0, x1_t), max(0, y1_t)
                        x2_t, y2_t = min(frame.shape[1], x2_t), min(frame.shape[0], y2_t)
                        if (x2_t - x1_t) > 10 and (y2_t - y1_t) > 10:
                            if len(target_templates) < 10:
                                if (time.time() - last_template_saved_time) > 0.5:
                                    crop_img = frame[y1_t:y2_t, x1_t:x2_t].copy()
                                    target_templates.append(crop_img)
                                    last_template_saved_time = time.time()

                    img_h, img_w = frame.shape[:2]
                    if self.lock_state in ["LOCKED", "REACQUISITION"]:
                        self.target_distance = ranging.estimate(target_track.track_id, w_box, h_box)
                        smoothed_pos, smoothed_vel = predictor.update(
                            target_track.track_id, cx, cy, self.target_distance, img_w, img_h
                        )
                    
                    pred_pos_3d = predictor.predict_future(steps=5)
                    if pred_pos_3d is not None:
                        predicted_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                    
                    if self.active_controller == "PN_GUIDANCE":
                        cmd = controller_pn.compute_commands(
                            target_track, self.target_distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                        )
                    elif self.active_controller == "FOLLOW_TARGET":
                        cmd = controller_follow.compute_commands(
                            target_track, self.target_distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                        )
                    else:
                        cmd = controller_pursuit.compute_commands(
                            target_track, self.target_distance, img_w, img_h
                        )
                    
                    self.current_cmd = cmd
                    if self.follow_target_enabled and self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                        mavlink_mgr.send_velocity_command(cmd['vx'], cmd['vy'], cmd['vz'], cmd['yaw_rate'])
                    elif self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                        mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                    
                    draw_target(frame, target_track, self.target_distance, predicted_px if self.debug_mode else None)
                    if self.lock_state == "LOST":
                        x1, y1, _, _ = map(int, target_track.tlbr)
                        cv2.putText(frame, "DEAD RECKONING ACTIVE", (x1, y1 - 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    self.current_cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
                    if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                        mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)

                # Warnings
                warnings = []
                if not mavlink_mgr.is_connected and not simulation_mode:
                    warnings.append("WARNING: MAVLINK DISCONNECTED")
                if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.current_mode != "GUIDED":
                    warnings.append("WARNING: VEHICLE NOT IN GUIDED MODE")
                if self.current_fps > 0 and self.current_fps < config['system']['min_fps_warning']:
                    warnings.append(f"WARNING: LOW LOOP FPS ({self.current_fps:.1f})")
                if target_track is not None and target_track.score < config['system']['low_confidence_warning']:
                    warnings.append(f"WARNING: LOW TARGET CONFIDENCE ({target_track.score:.2f})")
                if frame is not None and not virtual_target_mode and np.mean(frame) < 2.0:
                    warnings.append("WARNING: BLACK IMAGE - CHECK CAM INDEX")
                
                self.warnings_list = warnings

                # HUD
                draw_hud(
                    frame, 
                    (target_track is not None), 
                    self.current_fps, 
                    mavlink_mgr, 
                    self.active_controller, 
                    cmd, 
                    self.warnings_list,
                    target_id=self.current_target_id,
                    lock_status=self.lock_state,
                    follow_target_enabled=self.follow_target_enabled,
                    desired_distance=config['guidance'].get('desired_follow_distance', 5.0)
                )

                # Encode frame for web stream
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                self.latest_frame_encoded = buffer.tobytes()

                # Sync Telemetry status back
                self.telemetry = {
                    "is_connected": mavlink_mgr.is_connected,
                    "is_armed": mavlink_mgr.is_armed,
                    "current_mode": mavlink_mgr.current_mode,
                    "battery_voltage": round(mavlink_mgr.battery_voltage, 2),
                    "altitude": round(mavlink_mgr.altitude, 2),
                    "gps_lock": mavlink_mgr.gps_lock,
                    "roll": round(np.degrees(mavlink_mgr.roll), 1),
                    "pitch": round(np.degrees(mavlink_mgr.pitch), 1),
                    "yaw": round(np.degrees(mavlink_mgr.yaw), 1),
                    "local_x": round(mavlink_mgr.local_x, 2),
                    "local_y": round(mavlink_mgr.local_y, 2),
                    "local_z": round(mavlink_mgr.local_z, 2),
                }

                # FPS Calculation
                fps_frames += 1
                fps_accum += dt
                if fps_accum >= 1.0:
                    self.current_fps = fps_frames / fps_accum
                    fps_frames = 0
                    fps_accum = 0.0

                time.sleep(0.005)

        except Exception as e:
            logger.error(f"Error in pipeline run thread: {e}")
        finally:
            logger.info("Cleaning up pipeline thread resources...")
            video_stream.stop()
            if mavlink_mgr.is_connected:
                if mavlink_mgr.is_armed and not mavlink_mgr.simulation_mode:
                    mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                mavlink_mgr.disconnect()


# Instantiate Global Manager
pipeline_manager = PipelineManager()


# FastAPI REST Routes
@app.on_event("startup")
def startup_event():
    # Start tracking pipeline automatically
    pipeline_manager.start_pipeline()

@app.on_event("shutdown")
def shutdown_event():
    pipeline_manager.stop_pipeline()


@app.get("/api/status")
def get_status():
    return {
        "pipeline_running": pipeline_manager.running,
        "paused": pipeline_manager.paused,
        "guidance_active": pipeline_manager.guidance_active,
        "follow_target_enabled": pipeline_manager.follow_target_enabled,
        "active_controller": pipeline_manager.active_controller,
        "lock_state": pipeline_manager.lock_state,
        "target_id": pipeline_manager.current_target_id,
        "target_distance": round(pipeline_manager.target_distance, 2) if pipeline_manager.current_target_id else 0.0,
        "fps": round(pipeline_manager.current_fps, 1),
        "warnings": pipeline_manager.warnings_list,
        "telemetry": pipeline_manager.telemetry,
        "current_cmd": pipeline_manager.current_cmd,
        "system_mode": pipeline_manager.config.get('system', {}).get('mode', 'simulation')
    }


class SetModeRequest(BaseModel):
    mode: str

@app.post("/api/pipeline/set_mode")
def set_pipeline_mode(req: SetModeRequest):
    if req.mode not in ["simulation", "webcam", "hardware"]:
        raise HTTPException(status_code=400, detail="Invalid system mode")
        
    config = pipeline_manager.load_config()
    config['system']['mode'] = req.mode
    
    # Eventually switch the camera source based on selected mode
    if req.mode == "simulation":
        config['camera']['source'] = "0"
    elif req.mode == "webcam":
        config['camera']['source'] = "0"  # Local system webcam (usually index 0)
    elif req.mode == "hardware":
        # First hardware video interface on Linux, or external USB camera index 1 on Windows
        if os.name == 'posix':
            config['camera']['source'] = "/dev/video0"
        else:
            config['camera']['source'] = "1"  # Uses the external USB camera instead of built-in
            
    pipeline_manager.save_config(config)
    
    logger.info(f"System mode switched to '{req.mode}'. Restarting tracking pipeline...")
    pipeline_manager.stop_pipeline()
    time.sleep(1.0)
    success = pipeline_manager.start_pipeline()
    if success:
        return {"status": "success", "mode": req.mode, "message": f"Successfully switched to {req.mode} mode."}
    raise HTTPException(status_code=500, detail="Failed to restart tracking pipeline with new mode.")



# Connection controls
class TelemetryCommand(BaseModel):
    action: str  # arm, disarm, takeoff, land, rtl, emergency_stop
    altitude: Optional[float] = 3.0

@app.post("/api/control")
def post_control(cmd: TelemetryCommand):
    # Find MAVLink manager in running pipeline
    # Wait, we need to expose the control methods. We can just pipe them to mavlink_mgr in pipeline if it is running.
    # To keep it simple, we can run them on the connection manager or directly toggle states.
    logger.info(f"UI Command received: {cmd.action}")
    
    # We can write command interface functions in pipeline manager
    if not pipeline_manager.running:
        raise HTTPException(status_code=400, detail="Pipeline tracker is not running.")
        
    # Standard toggles
    if cmd.action == "toggle_guidance":
        pipeline_manager.guidance_active = not pipeline_manager.guidance_active
        return {"status": "success", "guidance_active": pipeline_manager.guidance_active}
        
    elif cmd.action == "toggle_follow":
        pipeline_manager.follow_target_enabled = not pipeline_manager.follow_target_enabled
        return {"status": "success", "follow_target_enabled": pipeline_manager.follow_target_enabled}
        
    elif cmd.action == "emergency_stop":
        pipeline_manager.guidance_active = False
        # We also need to send stop velocities immediately if MAVLink is active
        return {"status": "success", "guidance_active": False}
        
    # MAVLink physical commands
    # We will fetch the running mavlink_mgr instance using a trick or by attaching it to pipeline_manager
    # In _pipeline_loop, we can attach mavlink_mgr to self, let's make sure it is attached so we can access it here!
    # Let's check: in _pipeline_loop: "mavlink_mgr = MAVLinkConnectionManager(config)". We should do "self.mavlink_mgr = mavlink_mgr".
    # Wait, let's edit _pipeline_loop to use self.mavlink_mgr!
    # Let's look at pipeline_manager.mavlink_mgr.
    
    # Let's define the interface for mavlink commands:
    # We will modify the pipeline thread loop below to bind mavlink_mgr to self.mavlink_mgr.
    # Let's assume we do this in the code (which we will).
    
    # Let's trigger the action:
    # Actually, we can implement it cleanly:
    # Let's bind it in the runner
    pass


# We'll update the class methods to easily trigger MAVLink commands
class PipelineManager(PipelineManager):
    def run_mavlink_command(self, action: str, altitude: float = 3.0):
        if not self.running or not hasattr(self, 'mavlink_mgr') or self.mavlink_mgr is None:
            return False, "MAVLink connection not active."
            
        if action == "arm":
            res = self.mavlink_mgr.arm()
            return res, "Armed successfully" if res else "Failed to arm"
        elif action == "disarm":
            res = self.mavlink_mgr.disarm()
            return res, "Disarmed successfully" if res else "Failed to disarm"
        elif action == "takeoff":
            res = self.mavlink_mgr.takeoff(altitude_m=altitude)
            return res, f"Taking off to {altitude}m" if res else "Takeoff command failed"
        elif action == "land":
            res = self.mavlink_mgr.set_mode("LAND")
            return res, "Landing" if res else "Land command failed"
        elif action == "rtl":
            res = self.mavlink_mgr.set_mode("RTL")
            return res, "Returning to Launch" if res else "RTL command failed"
        return False, "Unknown action"


@app.post("/api/control/action")
def api_control_action(cmd: TelemetryCommand):
    success, msg = pipeline_manager.run_mavlink_command(cmd.action, cmd.altitude or 3.0)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@app.post("/api/control/toggle/{target}")
def api_toggle(target: str):
    if target == "guidance":
        pipeline_manager.guidance_active = not pipeline_manager.guidance_active
        return {"status": "success", "value": pipeline_manager.guidance_active}
    elif target == "follow":
        pipeline_manager.follow_target_enabled = not pipeline_manager.follow_target_enabled
        return {"status": "success", "value": pipeline_manager.follow_target_enabled}
    elif target == "debug":
        pipeline_manager.debug_mode = not pipeline_manager.debug_mode
        return {"status": "success", "value": pipeline_manager.debug_mode}
    elif target == "pause":
        pipeline_manager.paused = not pipeline_manager.paused
        return {"status": "success", "value": pipeline_manager.paused}
    else:
        raise HTTPException(status_code=400, detail="Invalid toggle target")


class ControllerCommand(BaseModel):
    controller: str # FOLLOW_TARGET, PN_GUIDANCE, DIRECT_PURSUIT

@app.post("/api/control/controller")
def api_set_controller(cmd: ControllerCommand):
    if cmd.controller in ["FOLLOW_TARGET", "PN_GUIDANCE", "DIRECT_PURSUIT"]:
        pipeline_manager.active_controller = cmd.controller
        return {"status": "success", "controller": pipeline_manager.active_controller}
    raise HTTPException(status_code=400, detail="Invalid controller type")


# Configuration endpoint
@app.get("/api/config")
def get_config():
    try:
        return pipeline_manager.load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read settings: {str(e)}")


@app.post("/api/config")
def update_config(new_config: Dict[str, Any]):
    try:
        pipeline_manager.save_config(new_config)
        return {"status": "success", "message": "Configuration updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@app.post("/api/pipeline/restart")
def restart_pipeline():
    logger.info("Restarting tracking pipeline...")
    pipeline_manager.stop_pipeline()
    time.sleep(1.0)
    success = pipeline_manager.start_pipeline()
    if success:
        return {"status": "success", "message": "Pipeline restarted successfully."}
    raise HTTPException(status_code=500, detail="Failed to restart tracking pipeline.")


# Video stream route
async def frame_generator():
    while True:
        if pipeline_manager.latest_frame_encoded is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + pipeline_manager.latest_frame_encoded + b'\r\n')
        # Rate limit stream to ~30 FPS
        await asyncio.sleep(0.033)

@app.get("/api/video_feed")
def get_video_feed():
    return StreamingResponse(
        frame_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# WebSocket for training subprocesses
@app.websocket("/api/ws/run_task")
async def websocket_run_task(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted for running task.")
    
    try:
        # Expect task parameter
        data = await websocket.receive_json()
        task_name = data.get("task")
        
        # Mapping names to scripts in 'coco json drone detection'
        scripts = {
            "preprocess": "restructure_custom_dataset.py",
            "augment": "augment_dataset.py",
            "enhance": "enhance_dataset.py",
            "auto_label": "auto_annotate.py",
            "train_strict": "train_strict_model.py",
            "train": "train_model.py",
            "evaluate": "evaluate.py",
            "strict_monitor": "strict_detector.py"
        }
        
        if task_name not in scripts:
            await websocket.send_json({"type": "error", "message": f"Unknown task: {task_name}"})
            await websocket.close()
            return
            
        script_file = scripts[task_name]
        cwd_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coco json drone detection")
        script_path = os.path.join(cwd_dir, script_file)
        
        if not os.path.exists(script_path):
            await websocket.send_json({"type": "error", "message": f"Script not found at: {script_path}"})
            await websocket.close()
            return

        await websocket.send_json({"type": "status", "message": f"Starting task: {task_name} ({script_file})..."})
        
        # Run process and pipe output
        # Use sys.executable to ensure we run with the correct Python interpreter
        # Send output line-by-line
        
        cmd = [sys.executable, "-u", script_file] # '-u' makes output unbuffered so logs stream instantly
        
        # Run process
        process = subprocess.Popen(
            cmd,
            cwd=cwd_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read lines asynchronously
        loop = asyncio.get_running_loop()
        
        def read_line():
            return process.stdout.readline()
            
        while True:
            # Run readline in executor to avoid blocking the async event loop
            line = await loop.run_in_executor(None, read_line)
            if not line:
                break
            
            # Send stdout line to websocket
            await websocket.send_json({"type": "log", "message": line.rstrip()})
            
        process.wait()
        exit_code = process.returncode
        
        if exit_code == 0:
            await websocket.send_json({"type": "done", "message": f"Task {task_name} completed successfully."})
        else:
            await websocket.send_json({"type": "error", "message": f"Task failed with exit code: {exit_code}"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# Correcting the _pipeline_loop to save local mavlink_mgr to self.mavlink_mgr
# by updating the actual code directly:
def _pipeline_loop_patch(self):
    config = self.load_config()
    config['system']['gui'] = False
    
    mode = config['system'].get('mode', 'simulation').lower()
    simulation_mode = (mode in ["simulation", "webcam"])
    config['system']['simulation_mode'] = simulation_mode

    # Initialize MAVLink Connection and bind to self
    mavlink_mgr = MAVLinkConnectionManager(config)
    mavlink_mgr.connect()
    
    with self.lock:
        self.mavlink_mgr = mavlink_mgr

    # Initialize core elements
    detector = YOLODetector(config)
    tracker = ByteTracker(config)
    ranging = DistanceEstimator(config)
    predictor = MotionPredictor(config)
    
    controller_pn = PNGuidanceController(config)
    controller_pursuit = DirectPursuitController(config)
    controller_follow = FollowTargetController(config)
    
    virtual_target_mode = (mode == "simulation")
    sitl_sim = None
    if virtual_target_mode:
        sitl_sim = SITLSimulator(config)

    # Video Capture
    video_stream = ThreadedVideoStream(config)
    video_stream.start()

    target_templates = []
    last_template_saved_time = 0.0
    self.lock_state = "ACQUISITION"
    lost_time = 0.0
    self.last_known_box = None
    self.current_target_id = None
    self.target_distance = 0.0
    
    lk_tracker = BoundingBoxTracker()
    missing_yolo_frames = 0
    
    last_frame_time = time.time()
    fps_accum = 0
    fps_frames = 0

    logger.info("Pipeline processing loop active.")

    try:
        while self.running:
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            
            if self.paused:
                time.sleep(0.03)
                continue

            grabbed, frame = video_stream.read(wait=(not virtual_target_mode), timeout=0.1)
            if not grabbed and not virtual_target_mode:
                if video_stream.stopped:
                    logger.error("Video stream stopped.")
                    break
                time.sleep(0.01)
                continue

            if frame is None and virtual_target_mode:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            lk_updated = False
            lk_box = None
            if lk_tracker.active:
                lk_updated, lk_box = lk_tracker.update_tracker(frame_gray)

            detections = []
            if virtual_target_mode:
                interceptor_pos = np.array([mavlink_mgr.local_x, mavlink_mgr.local_y, mavlink_mgr.local_z])
                interceptor_att = np.array([mavlink_mgr.roll, mavlink_mgr.pitch, mavlink_mgr.yaw])
                img_h, img_w = frame.shape[:2]
                sim_detection, rel_3d = sitl_sim.project_target_to_camera(
                    interceptor_pos, interceptor_att, img_w, img_h
                )
                frame = sitl_sim.generate_virtual_frame(sim_detection, img_w, img_h)
                if sim_detection is not None:
                    detections = [sim_detection]
            else:
                active_conf = max(0.20, config['yolo']['confidence_threshold'] - 0.10) if lk_tracker.active else config['yolo']['confidence_threshold']
                detections = detector.detect(frame, conf_threshold=active_conf)

            active_tracks = tracker.update(detections)
            target_track = None
            from_yolo = False

            pred_cx, pred_cy, pred_w, pred_h = None, None, None, None
            if self.current_target_id is not None:
                pred_pos_3d = predictor.state[:3]
                img_h, img_w = frame.shape[:2]
                pred_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                if pred_px is not None and predictor.covariance[0,0] < 5000:
                    pred_cx, pred_cy = pred_px
                    pred_w = (ranging.focal_length * ranging.drone_real_size) / max(0.1, pred_pos_3d[2])
                    pred_h = pred_w
                elif self.last_known_box is not None:
                    pred_cx, pred_cy, pred_w, pred_h = self.last_known_box

            matched_track = None
            matched_det = None
            best_assoc_score = 0.0

            if self.current_target_id is not None:
                for track in active_tracks:
                    txc, tyc, tw, th = track.xywh
                    iou_pred = calculate_iou([txc, tyc, tw, th], [pred_cx, pred_cy, pred_w, pred_h]) if pred_cx is not None else 0.0
                    iou_lk = calculate_iou([txc, tyc, tw, th], [lk_box[0] + (lk_box[2]-lk_box[0])/2, lk_box[1] + (lk_box[3]-lk_box[1])/2, lk_box[2]-lk_box[0], lk_box[3]-lk_box[1]]) if (lk_tracker.active and lk_updated and lk_box is not None) else 0.0
                    dist_score = 0.0
                    if pred_cx is not None:
                        dist = np.hypot(txc - pred_cx, tyc - pred_cy)
                        max_dist = max(300.0, frame.shape[1] * 0.5)
                        if dist < max_dist:
                            dist_score = 1.0 - (dist / max_dist)
                    assoc_score = max(iou_pred, iou_lk, dist_score * 0.5)
                    if assoc_score > 0.20 and assoc_score > best_assoc_score:
                        best_assoc_score = assoc_score
                        matched_track = track
                        matched_det = None
                        
                if matched_track is None:
                    for det in detections:
                        dx1, dy1, dx2, dy2, conf = det[:5]
                        dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                        dw, dh = dx2 - dx1, dy2 - dy1
                        iou_pred = calculate_iou([dxc, dyc, dw, dh], [pred_cx, pred_cy, pred_w, pred_h]) if pred_cx is not None else 0.0
                        iou_lk = calculate_iou([dxc, dyc, dw, dh], [lk_box[0] + (lk_box[2]-lk_box[0])/2, lk_box[1] + (lk_box[3]-lk_box[1])/2, lk_box[2]-lk_box[0], lk_box[3]-lk_box[1]]) if (lk_tracker.active and lk_updated and lk_box is not None) else 0.0
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

            reid_matched_track = None
            reid_matched_det = None
            best_sim_score = 0.0
            if len(target_templates) > 0:
                for track in active_tracks:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop = frame[y1:y2, x1:x2]
                        for temp in target_templates:
                            sim = compute_template_similarity(crop, temp)
                            if sim > best_sim_score:
                                best_sim_score = sim
                                reid_matched_track = track
                                reid_matched_det = None
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop = frame[y1:y2, x1:x2]
                        for temp in target_templates:
                            sim = compute_template_similarity(crop, temp)
                            if sim > best_sim_score:
                                best_sim_score = sim
                                reid_matched_track = None
                                reid_matched_det = det

            if best_sim_score >= 0.70:
                if reid_matched_track is not None:
                    if self.current_target_id is None:
                        self.current_target_id = reid_matched_track.track_id
                    reid_matched_track.track_id = self.current_target_id
                    matched_track = reid_matched_track
                elif reid_matched_det is not None:
                    if self.current_target_id is None:
                        self.current_target_id = 99
                    matched_det = reid_matched_det

            if self.current_target_id is not None and matched_track is None and matched_det is None:
                if len(active_tracks) == 1:
                    matched_track = active_tracks[0]
                elif len(detections) == 1:
                    matched_det = detections[0]

            if matched_track is not None:
                matched_track.track_id = self.current_target_id
                target_track = matched_track
                from_yolo = True
            elif matched_det is not None:
                target_track = MockLKTrack(self.current_target_id, matched_det[:4], score=matched_det[4])
                from_yolo = True
            elif self.current_target_id is not None and lk_tracker.active and lk_updated and lk_box is not None:
                missing_yolo_frames += 1
                lk_max_frames = config['tracker'].get('lk_max_fallback_frames', 15)
                if missing_yolo_frames <= lk_max_frames:
                    lk_tracker.last_conf *= 0.95
                    target_track = MockLKTrack(self.current_target_id, lk_box, score=lk_tracker.last_conf)
                    from_yolo = False
                else:
                    lk_tracker.active = False
                    target_track = None
            else:
                if self.current_target_id is not None:
                    lk_tracker.active = False

            if target_track is None and self.current_target_id is not None and self.lock_state == "LOST":
                best_cand = None
                if len(active_tracks) > 0:
                    active_tracks.sort(key=lambda x: x.score, reverse=True)
                    if active_tracks[0].score >= config['tracker']['track_threshold']:
                        best_cand = active_tracks[0]
                        best_cand.track_id = self.current_target_id
                        target_track = best_cand
                        from_yolo = True
                if best_cand is None and len(detections) > 0:
                    detections_sort = sorted(detections, key=lambda x: x[4], reverse=True)
                    if detections_sort[0][4] >= config['tracker']['track_threshold']:
                        best_det = detections_sort[0]
                        target_track = MockLKTrack(self.current_target_id, best_det[:4], score=best_det[4])
                        from_yolo = True

            if target_track is not None:
                if self.lock_state == "ACQUISITION":
                    self.lock_state = "LOCKED"
                elif self.lock_state == "LOST":
                    self.lock_state = "REACQUISITION"
                elif self.lock_state == "REACQUISITION":
                    self.lock_state = "LOCKED"
                
                if from_yolo:
                    x1, y1, x2, y2 = target_track.tlbr
                    lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                    lk_tracker.last_conf = target_track.score
                    missing_yolo_frames = 0
            else:
                if self.current_target_id is not None:
                    if self.lock_state in ["LOCKED", "REACQUISITION"]:
                        self.lock_state = "LOST"
                        lost_time = time.time()
                    
                    elapsed_lost = time.time() - lost_time
                    reacq_timeout = config['guidance'].get('reacquisition_timeout', 5.0)
                    loss_timeout = config['guidance'].get('target_loss_timeout', 3.0)
                    
                    if elapsed_lost > reacq_timeout:
                        self.lock_state = "ACQUISITION"
                        self.current_target_id = None
                        self.last_known_box = None
                        predictor.reset()
                        controller_pn.reset()
                        controller_pursuit.reset()
                        controller_follow.reset()
                        if config['guidance'].get('rtl_after_timeout', False) and mavlink_mgr.is_connected:
                            mavlink_mgr.set_mode("RTL")
                    elif elapsed_lost > loss_timeout:
                        if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                            mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                    else:
                        predictor.state = np.dot(predictor.F, predictor.state)
                        predictor.covariance = np.dot(predictor.F, np.dot(predictor.covariance, predictor.F.T)) + predictor.Q
                        smoothed_pos = predictor.state[:3]
                        img_h, img_w = frame.shape[:2]
                        pred_px = predictor.project_to_image(smoothed_pos, img_w, img_h)
                        if pred_px is not None:
                            cx_p, cy_p = pred_px
                            box_sz = (ranging.focal_length * ranging.drone_real_size) / max(0.1, smoothed_pos[2])
                            x1_p, y1_p = cx_p - box_sz / 2.0, cy_p - box_sz / 2.0
                            x2_p, y2_p = cx_p + box_sz / 2.0, cy_p + box_sz / 2.0
                            target_track = MockLKTrack(self.current_target_id, [x1_p, y1_p, x2_p, y2_p], score=0.4)
                            self.target_distance = smoothed_pos[2]
                else:
                    if len(active_tracks) > 0:
                        active_tracks.sort(key=lambda x: x.score, reverse=True)
                        best_cand = active_tracks[0]
                        if best_cand.score >= config['tracker']['track_threshold']:
                            target_track = best_cand
                            self.current_target_id = target_track.track_id
                            self.lock_state = "LOCKED"
                            x1, y1, x2, y2 = target_track.tlbr
                            lk_tracker.init_tracker(frame_gray, [x1, y1, x2, y2])
                            lk_tracker.last_conf = target_track.score
                            missing_yolo_frames = 0
                            predictor.reset()
                            controller_pn.reset()
                            controller_pursuit.reset()
                            controller_follow.reset()

            cmd = None
            predicted_px = None

            if target_track is not None:
                cx, cy, w_box, h_box = target_track.xywh
                self.last_known_box = [cx, cy, w_box, h_box]
                
                if self.follow_target_enabled and self.lock_state in ["LOCKED", "REACQUISITION"] and from_yolo:
                    x1_t, y1_t, x2_t, y2_t = map(int, target_track.tlbr)
                    x1_t, y1_t = max(0, x1_t), max(0, y1_t)
                    x2_t, y2_t = min(frame.shape[1], x2_t), min(frame.shape[0], y2_t)
                    if (x2_t - x1_t) > 10 and (y2_t - y1_t) > 10:
                        if len(target_templates) < 10:
                            if (time.time() - last_template_saved_time) > 0.5:
                                crop_img = frame[y1_t:y2_t, x1_t:x2_t].copy()
                                target_templates.append(crop_img)
                                last_template_saved_time = time.time()

                img_h, img_w = frame.shape[:2]
                if self.lock_state in ["LOCKED", "REACQUISITION"]:
                    self.target_distance = ranging.estimate(target_track.track_id, w_box, h_box)
                    smoothed_pos, smoothed_vel = predictor.update(
                        target_track.track_id, cx, cy, self.target_distance, img_w, img_h
                    )
                
                pred_pos_3d = predictor.predict_future(steps=5)
                if pred_pos_3d is not None:
                    predicted_px = predictor.project_to_image(pred_pos_3d, img_w, img_h)
                
                if self.active_controller == "PN_GUIDANCE":
                    cmd = controller_pn.compute_commands(
                        target_track, self.target_distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                    )
                elif self.active_controller == "FOLLOW_TARGET":
                    cmd = controller_follow.compute_commands(
                        target_track, self.target_distance, smoothed_pos, smoothed_vel, img_w, img_h, dt
                    )
                else:
                    cmd = controller_pursuit.compute_commands(
                        target_track, self.target_distance, img_w, img_h
                    )
                
                self.current_cmd = cmd
                if self.follow_target_enabled and self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(cmd['vx'], cmd['vy'], cmd['vz'], cmd['yaw_rate'])
                elif self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                
                draw_target(frame, target_track, self.target_distance, predicted_px if self.debug_mode else None)
                if self.lock_state == "LOST":
                    x1, y1, _, _ = map(int, target_track.tlbr)
                    cv2.putText(frame, "DEAD RECKONING ACTIVE", (x1, y1 - 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                self.current_cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
                if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.is_armed:
                    mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)

            warnings = []
            if not mavlink_mgr.is_connected and not simulation_mode:
                warnings.append("WARNING: MAVLINK DISCONNECTED")
            if self.guidance_active and mavlink_mgr.is_connected and mavlink_mgr.current_mode != "GUIDED":
                warnings.append("WARNING: VEHICLE NOT IN GUIDED MODE")
            if self.current_fps > 0 and self.current_fps < config['system']['min_fps_warning']:
                warnings.append(f"WARNING: LOW LOOP FPS ({self.current_fps:.1f})")
            if target_track is not None and target_track.score < config['system']['low_confidence_warning']:
                warnings.append(f"WARNING: LOW TARGET CONFIDENCE ({target_track.score:.2f})")
            if frame is not None and not virtual_target_mode and np.mean(frame) < 2.0:
                warnings.append("WARNING: BLACK IMAGE - CHECK CAM INDEX")
            
            self.warnings_list = warnings

            draw_hud(
                frame, 
                (target_track is not None), 
                self.current_fps, 
                mavlink_mgr, 
                self.active_controller, 
                cmd, 
                self.warnings_list,
                target_id=self.current_target_id,
                lock_status=self.lock_state,
                follow_target_enabled=self.follow_target_enabled,
                desired_distance=config['guidance'].get('desired_follow_distance', 5.0)
            )

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self.latest_frame_encoded = buffer.tobytes()

            with self.lock:
                self.telemetry = {
                    "is_connected": mavlink_mgr.is_connected,
                    "is_armed": mavlink_mgr.is_armed,
                    "current_mode": mavlink_mgr.current_mode,
                    "battery_voltage": round(mavlink_mgr.battery_voltage, 2),
                    "altitude": round(mavlink_mgr.altitude, 2),
                    "gps_lock": mavlink_mgr.gps_lock,
                    "roll": round(np.degrees(mavlink_mgr.roll), 1),
                    "pitch": round(np.degrees(mavlink_mgr.pitch), 1),
                    "yaw": round(np.degrees(mavlink_mgr.yaw), 1),
                    "local_x": round(mavlink_mgr.local_x, 2),
                    "local_y": round(mavlink_mgr.local_y, 2),
                    "local_z": round(mavlink_mgr.local_z, 2),
                }

            fps_frames += 1
            fps_accum += dt
            if fps_accum >= 1.0:
                self.current_fps = fps_frames / fps_accum
                fps_frames = 0
                fps_accum = 0.0
            # No sleep — let YOLO inference time naturally rate-limit the loop
            # for zero artificial latency.

    except Exception as e:
        logger.error(f"Error in pipeline run thread: {e}")
    finally:
        logger.info("Cleaning up pipeline thread resources...")
        video_stream.stop()
        if mavlink_mgr.is_connected:
            if mavlink_mgr.is_armed and not mavlink_mgr.simulation_mode:
                mavlink_mgr.send_velocity_command(0.0, 0.0, 0.0, 0.0)
            mavlink_mgr.disconnect()
        with self.lock:
            self.mavlink_mgr = None

# Bind patched method
PipelineManager._pipeline_loop = _pipeline_loop_patch


if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
