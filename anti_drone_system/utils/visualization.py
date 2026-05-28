import cv2
import numpy as np

def draw_hud(frame, tracking_active, current_fps, mavlink_mgr, active_controller_name, cmd_dict=None, warnings=[]):
    """
    Draws an advanced tactical Head-Up Display (HUD) overlay on the frame.
    Supports SpeedyBee and ArduPilot telemetry displays.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw central targeting reticle
    color_reticle = (0, 255, 0) if tracking_active else (0, 165, 255)
    cv2.drawMarker(frame, (cx, cy), color_reticle, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1)
    cv2.circle(frame, (cx, cy), 40, color_reticle, thickness=1, lineType=cv2.LINE_AA)

    # 1. Top-Left: System Status & Metrics
    overlay_y = 30
    cv2.putText(frame, "ANTI-DRONE INTERCEPTION SYSTEM", (20, overlay_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    overlay_y += 25
    status_text = "TARGET STATE: LOCKED" if tracking_active else "TARGET STATE: ACQUIRING"
    status_color = (0, 255, 0) if tracking_active else (0, 165, 255)
    cv2.putText(frame, status_text, (20, overlay_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    overlay_y += 20
    cv2.putText(frame, f"LOOP RATE: {current_fps:.1f} FPS", (20, overlay_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display active guidance law
    overlay_y += 20
    cv2.putText(frame, f"CTRL LAW: {active_controller_name}", (20, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # 2. Top-Right: SpeedyBee / ArduPilot Telemetry
    telemetry_y = 30
    sim_indicator = " (SITL)" if mavlink_mgr.simulation_mode else " (SPEEDYBEE)"
    conn_text = f"MAVLINK: CONNECTED{sim_indicator}" if mavlink_mgr.is_connected else "MAVLINK: RECONNECTING..."
    conn_color = (0, 255, 0) if mavlink_mgr.is_connected else (0, 0, 255)
    cv2.putText(frame, conn_text, (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1, cv2.LINE_AA)

    telemetry_y += 20
    arm_text = "ARMED" if mavlink_mgr.is_armed else "DISARMED"
    arm_color = (0, 255, 0) if mavlink_mgr.is_armed else (0, 0, 255)
    cv2.putText(frame, f"STATE: {arm_text}", (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1, cv2.LINE_AA)

    telemetry_y += 20
    cv2.putText(frame, f"MODE: {mavlink_mgr.current_mode}", (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    telemetry_y += 20
    cv2.putText(frame, f"BATTERY: {mavlink_mgr.battery_voltage:.2f} V", (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    telemetry_y += 20
    cv2.putText(frame, f"ALTITUDE: {mavlink_mgr.altitude:.1f} m", (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    telemetry_y += 20
    gps_color = (0, 255, 0) if "3D" in mavlink_mgr.gps_lock else (0, 165, 255)
    cv2.putText(frame, f"GPS LOCK: {mavlink_mgr.gps_lock}", (w - 300, telemetry_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 1, cv2.LINE_AA)

    # 3. Bottom-Left: Computed Commands
    if cmd_dict:
        cmd_y = h - 130
        cv2.putText(frame, "GUIDANCE VELOCITY TARGETS:", (20, cmd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        
        cmd_y += 20
        cv2.putText(frame, f"Vx (Forward): {cmd_dict['vx']:.2f} m/s", (20, cmd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cmd_y += 20
        cv2.putText(frame, f"Vy (Lateral): {cmd_dict['vy']:.2f} m/s", (20, cmd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cmd_y += 20
        cv2.putText(frame, f"Vz (Vertical): {cmd_dict['vz']:.2f} m/s", (20, cmd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cmd_y += 20
        cv2.putText(frame, f"Yaw Rate: {cmd_dict['yaw_rate']:.2f} rad/s", (20, cmd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Bottom-Right: Target Lock Banner
        lock_y = h - 60
        lock_text = "INTERCEPT LOCK" if cmd_dict['aligned'] else "ALIGNING"
        lock_color = (0, 255, 0) if cmd_dict['aligned'] else (0, 165, 255)
        cv2.rectangle(frame, (w - 200, lock_y - 20), (w - 20, lock_y + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 200, lock_y - 20), (w - 20, lock_y + 10), lock_color, 1)
        cv2.putText(frame, lock_text, (w - 185, lock_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lock_color, 1, cv2.LINE_AA)

    # 5. Top-Center: Safety Alerts / Warnings Banner
    if warnings:
        warn_y = 60
        for warn in warnings:
            # Render warning banners
            text_size = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            start_x = cx - text_size[0] // 2
            cv2.rectangle(frame, (start_x - 10, warn_y - 18), (start_x + text_size[0] + 10, warn_y + 6), (0, 0, 150), -1)
            cv2.putText(frame, warn, (start_x, warn_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            warn_y += 28


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
