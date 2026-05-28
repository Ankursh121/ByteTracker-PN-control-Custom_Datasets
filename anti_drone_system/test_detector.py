import cv2
import time
from ultralytics import YOLO
import os

def test():
    model_path = r"d:\Bytetrack\anti_drone_system\models\best.pt"
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
        
    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera index 0.")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Failed to open camera index 1.")
            return

    print("Camera opened. Capturing 20 frames and running inference...")
    time.sleep(1.0) # Let camera warm up
    
    for i in range(20):
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            time.sleep(0.1)
            continue
            
        results = model.predict(frame, conf=0.10, verbose=False)
        print(f"\nFrame {i}:")
        if len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) == 0:
                print("  No detections.")
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                print(f"  Class: {cls}, Conf: {conf:.4f}, Box: {xyxy}")
        else:
            print("  No results.")
        time.sleep(0.1)
        
    cap.release()
    print("Test finished.")

if __name__ == "__main__":
    test()
