import os
import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger("AntiDroneSystem.Detector")

class YOLODetector:
    def __init__(self, config):
        """
        Initializes the YOLO detector with settings from the configuration.
        """
        self.config = config
        self.model_path = config['yolo']['model_path']
        self.conf_threshold = config['yolo']['confidence_threshold']
        self.nms_threshold = config['yolo']['nms_threshold']
        self.imgsz = config['yolo']['imgsz']
        self.device = config['yolo']['device']
        self.target_class_ids = config['yolo']['target_class_ids']
        self.max_detections = config['yolo']['max_detections']

        # Determine and validate device configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Load weights, with safety fallback for local testing
        if not os.path.exists(self.model_path):
            logger.warning(f"Trained model weights not found at: {self.model_path}")
            parent_fallback = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "yolov8n.pt")
            if os.path.exists(parent_fallback):
                self.model_path = parent_fallback
            else:
                self.model_path = "yolov8n.pt"
            logger.warning(f"Falling back to standard pre-trained model: {self.model_path}")
            
            # Ensure the models directory exists
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
        
        logger.info(f"Loading YOLO model weights from {self.model_path} on device {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            # Send model to device
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully.")
            
            # GPU Warmup to eliminate first-frame inference lag
            if "cuda" in str(self.device):
                import numpy as np
                logger.info("Enabling cuDNN benchmark optimization...")
                torch.backends.cudnn.benchmark = True
                logger.info("Warming up YOLO model on CUDA...")
                warmup_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                _ = self.model.predict(
                    source=warmup_img, 
                    imgsz=self.imgsz, 
                    device=self.device, 
                    verbose=False, 
                    half=True
                )
                logger.info("YOLO model warmup complete.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, frame, conf_threshold=None):
        """
        Runs YOLO inference on a single frame.
        Returns:
            list of detections: [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        use_half = "cuda" in str(self.device)
        
        # Run inference under torch.no_grad to speed up and reduce memory
        with torch.no_grad():
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=self.nms_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_detections,
                classes=self.target_class_ids,
                verbose=False,
                half=use_half
            )

        # Get frame dimensions and calculate maximum size limits
        h_frame, w_frame = frame.shape[:2]
        frame_area = w_frame * h_frame
        max_area = frame_area * self.config['yolo'].get('max_box_ratio', 0.50)
        max_w = w_frame * self.config['yolo'].get('max_box_width_ratio', 0.80)
        max_h = h_frame * self.config['yolo'].get('max_box_height_ratio', 0.80)

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if len(boxes) > 0:
                # Retrieve all attributes to CPU/numpy at once to prevent costly GPU-CPU synchronizations inside the loop
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    xyxy = xyxys[i]
                    conf_val = float(confs[i])
                    cls_id = int(clss[i])
                    
                    # Bounding box dimensions
                    w_box = xyxy[2] - xyxy[0]
                    h_box = xyxy[3] - xyxy[1]
                    box_area = w_box * h_box
                    
                    # Filter out detections that exceed size limits (likely false positives)
                    if box_area > max_area or w_box > max_w or h_box > max_h:
                        logger.debug(
                            f"Filtered large detection box: Box={[int(c) for c in xyxy]}, "
                            f"Area={box_area:.1f} (max={max_area:.1f}), "
                            f"W={w_box:.1f} (max={max_w:.1f}), H={h_box:.1f} (max={max_h:.1f})"
                        )
                        continue
                    
                    detections.append([
                        xyxy[0],  # x1
                        xyxy[1],  # y1
                        xyxy[2],  # x2
                        xyxy[3],  # y2
                        conf_val,
                        cls_id
                    ])
                
        return detections
