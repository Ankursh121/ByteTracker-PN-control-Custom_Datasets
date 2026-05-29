import cv2
import numpy as np

def draw_hud(frame, tracking_active, current_fps, mavlink_mgr, active_controller_name, cmd_dict=None, warnings=[], target_id=None, lock_status="ACQUISITION", follow_target_enabled=False, desired_distance=5.0):
    """
    Draws a minimal tactical reticle (crosshair + circle) on the frame.
    All status text is rendered in the React web UI instead.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Color mapping for different lock states
    state_colors = {
        "ACQUISITION": (0, 0, 255),
        "LOCKED": (0, 255, 0),
        "LOST": (0, 255, 255),
        "REACQUISITION": (0, 165, 255)
    }
    color_reticle = state_colors.get(lock_status, (0, 165, 255))

    # Draw central targeting reticle only
    cv2.drawMarker(frame, (cx, cy), color_reticle, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1)
    cv2.circle(frame, (cx, cy), 40, color_reticle, thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (cx - 25, cy - 20), (cx + 25, cy + 20), (50, 50, 50), 1, cv2.LINE_AA)



def draw_target(frame, track, distance, predicted_px=None):
    """
    Draws bounding box details and 3D prediction projections.
    """
    x1, y1, x2, y2 = map(int, track.tlbr)
    track_id = track.track_id
    conf = track.score

    color_box = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2, cv2.LINE_AA)

    # Corners overlay
    length = min(15, int((x2 - x1) * 0.2))
    cv2.line(frame, (x1, y1), (x1 + length, y1), color_box, 4)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color_box, 4)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color_box, 4)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color_box, 4)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color_box, 4)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color_box, 4)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color_box, 4)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color_box, 4)

    # Print labels
    label = f"ID: {track_id} | {conf:.2f}"
    dist_label = f"RNG: {distance:.2f}m"

    cv2.rectangle(frame, (x1, y1 - 38), (x1 + 140, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, dist_label, (x1 + 5, y1 - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if predicted_px:
        pred_x, pred_y = predicted_px
        cv2.circle(frame, (pred_x, pred_y), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (pred_x, pred_y), (0, 255, 255), 1, cv2.LINE_4)
        cv2.putText(frame, "PRED INTERCEPT", (pred_x + 10, pred_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
