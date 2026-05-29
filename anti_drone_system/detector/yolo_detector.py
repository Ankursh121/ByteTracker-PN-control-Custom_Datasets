import os
import logging
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("AntiDroneSystem.Detector")


class YOLODetector:
    def __init__(self, config):
        """
        Initializes the YOLO detector with settings from the configuration.

        Two operating modes depending on which weights are loaded:

        1. CUSTOM MODEL (models/best.pt):
           - class 0 = drone (your trained dataset)
           - target_class_ids: [0] filters directly to drone class
           - No COCO class blocking needed

        2. FALLBACK MODEL (yolov8n.pt, COCO 80 classes):
           - class 0 = person (wrong!)
           - target_class_ids is overridden to None (detect all classes)
           - fallback_blocked_class_ids blocks person/bird/cat/dog
           - Aspect ratio filter additionally rejects person-shaped boxes
        """
        self.config = config
        self.model_path = config['yolo']['model_path']
        self.conf_threshold = config['yolo']['confidence_threshold']
        self.nms_threshold = config['yolo']['nms_threshold']
        self.imgsz = config['yolo']['imgsz']
        self.device = config['yolo']['device']
        self.max_detections = config['yolo']['max_detections']
        self.is_fallback_model = False

        # Determine and validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        # Resolve model path relative to the anti_drone_system package directory
        # so it works regardless of which directory the server is launched from
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # anti_drone_system/
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(_pkg_dir, self.model_path)

        # Load weights — custom model first, COCO fallback if missing
        if not os.path.exists(self.model_path):
            logger.warning(f"Custom model not found at: {self.model_path}")
            parent_fallback = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "yolov8n.pt"
            )
            if os.path.exists(parent_fallback):
                self.model_path = parent_fallback
            else:
                self.model_path = "yolov8n.pt"

            self.is_fallback_model = True
            logger.warning(
                f"Using COCO fallback model: {self.model_path}\n"
                "  → target_class_ids overridden to None (COCO class 0 = person, not drone)\n"
                "  → fallback_blocked_class_ids and aspect ratio filter will be applied"
            )
            os.makedirs(
                os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".",
                exist_ok=True
            )
        else:
            logger.info(f"Custom drone model found: {self.model_path}")

        # Class filtering — depends on which model is loaded
        if self.is_fallback_model:
            # COCO model: do NOT restrict to class 0 (that's person!)
            # Let all classes through; shape/blocklist filters handle rejection
            self.target_class_ids = None
            self.blocked_class_ids = set(config['yolo'].get('fallback_blocked_class_ids', [0, 14, 15, 16]))
            self.min_aspect_ratio = config['yolo'].get('fallback_min_aspect_ratio', 0.55)
            self.max_aspect_ratio = config['yolo'].get('fallback_max_aspect_ratio', 2.50)
        else:
            # Custom drone model: class 0 = drone — use it directly
            self.target_class_ids = config['yolo'].get('target_class_ids', [0])
            self.blocked_class_ids = set()  # custom model has no COCO classes to block
            self.min_aspect_ratio = None    # no aspect ratio restriction needed
            self.max_aspect_ratio = None

        logger.info(
            f"Detector init: model={'FALLBACK' if self.is_fallback_model else 'CUSTOM'}, "
            f"target_class_ids={self.target_class_ids}, "
            f"blocked={self.blocked_class_ids}, "
            f"conf={self.conf_threshold}, imgsz={self.imgsz}, device={self.device}"
        )

        logger.info(f"Loading YOLO weights from {self.model_path} on {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully.")

            # CUDA warmup — eliminates first-frame inference spike
            if "cuda" in str(self.device):
                logger.info("Enabling cuDNN benchmark mode...")
                torch.backends.cudnn.benchmark = True
                logger.info("Warming up YOLO on CUDA...")
                warmup_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                _ = self.model.predict(
                    source=warmup_img,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    half=True
                )
                logger.info("YOLO warmup complete.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame, conf_threshold=None):
        """
        Runs YOLO inference on a single frame.

        Custom model path:  class filter → size filter → return
        Fallback COCO path: class blocklist + aspect ratio + size filter → return

        Returns:
            list: [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        use_half = "cuda" in str(self.device)

        with torch.no_grad():
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=self.nms_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_detections,
                classes=self.target_class_ids,  # None = all; [0] = drone class on custom model
                verbose=False,
                half=use_half
            )

        # Frame size limits (applied to ALL detections)
        h_frame, w_frame = frame.shape[:2]
        frame_area = w_frame * h_frame
        max_area = frame_area * self.config['yolo'].get('max_box_ratio', 0.50)
        max_w    = w_frame   * self.config['yolo'].get('max_box_width_ratio', 0.80)
        max_h    = h_frame   * self.config['yolo'].get('max_box_height_ratio', 0.75)
        min_area = self.config['tracker'].get('min_box_area', 50)

        detections = []

        if not results or len(results[0].boxes) == 0:
            return detections

        # Bulk GPU→CPU transfer (one sync, not per-box)
        boxes  = results[0].boxes
        xyxys  = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            cls_id   = clss[i]
            conf_val = float(confs[i])
            xyxy     = xyxys[i]

            # ── FALLBACK-ONLY: block known COCO non-drone classes ─────────
            if self.blocked_class_ids and cls_id in self.blocked_class_ids:
                logger.debug(f"[Fallback] Blocked COCO class {cls_id}")
                continue

            w_box    = xyxy[2] - xyxy[0]
            h_box    = xyxy[3] - xyxy[1]
            box_area = w_box * h_box

            # ── Min area gate (noise rejection) ───────────────────────────
            if box_area < min_area:
                logger.debug(f"Tiny box filtered: area={box_area:.0f}")
                continue

            # ── Max size gate (too-large box = background/person nearby) ──
            if box_area > max_area or w_box > max_w or h_box > max_h:
                logger.debug(
                    f"Oversized box filtered: cls={cls_id} conf={conf_val:.2f} "
                    f"area={box_area:.0f}/{max_area:.0f}"
                )
                continue

            # ── FALLBACK-ONLY: aspect ratio (persons are tall, drones compact)
            if self.min_aspect_ratio is not None and h_box > 0:
                ar = w_box / h_box
                if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                    logger.debug(
                        f"[Fallback] AR filtered: cls={cls_id} conf={conf_val:.2f} "
                        f"AR={ar:.2f} valid=[{self.min_aspect_ratio:.2f},{self.max_aspect_ratio:.2f}]"
                    )
                    continue

            detections.append([
                xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                conf_val,
                cls_id
            ])

        return detections
