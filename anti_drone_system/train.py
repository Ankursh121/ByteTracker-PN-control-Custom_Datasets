import os
import shutil
import logging
import torch
from ultralytics import YOLO
import yaml

from utils.dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("AntiDroneSystem.Train")

def load_settings():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "settings.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Settings file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training():
    logger.info("Initializing YOLO Drone Training Pipeline...")
    
    # 1. Dataset Setup & Validation
    manager = DatasetManager()
    summary = manager.validate_and_reconstruct()
    
    # 2. Settings parsing
    config = load_settings()
    
    # Extract training configs (with safe defaults)
    yolo_cfg = config.get('yolo', {})
    train_cfg = yolo_cfg.get('train_settings', {})
    epochs = train_cfg.get('epochs', 50)  # Standard epochs is 50 for good convergence
    
    # 3. Hardware detection and batch sizing
    if torch.cuda.is_available():
        device = "0"
        amp = True
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) # GB
        logger.info(f"CUDA GPU detected: '{torch.cuda.get_device_name(0)}' with {total_memory:.2f} GB VRAM.")
        
        # Optimize batch size based on VRAM
        if total_memory >= 10.0:
            batch_size = 32
            workers = 8
            logger.info("Setting training parameters: High-Performance (Batch=32, Workers=8).")
        elif total_memory >= 6.0:
            batch_size = 16
            workers = 4
            logger.info("Setting training parameters: Standard GPU (Batch=16, Workers=4).")
        else:
            batch_size = 8
            workers = 2
            logger.info("Setting training parameters: Low VRAM (Batch=8, Workers=2).")
    else:
        device = "cpu"
        amp = False
        batch_size = 2
        workers = 0
        logger.warning("CUDA GPU not available. Training will run in CPU fallback mode (Batch=2, Workers=0).")
        
    # Search for yolov8s.pt locally to avoid redownloads
    base_model_path = "yolov8s.pt"
    search_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8s.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coco json drone detection", "yolov8s.pt"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            base_model_path = path
            logger.info(f"Found local base model at: {base_model_path}")
            break
            
    if base_model_path == "yolov8s.pt":
        logger.info("Local base model not found. YOLO will download yolov8s.pt from Ultralytics server.")
        
    model = YOLO(base_model_path)
    
    data_yaml_path = os.path.join(manager.target_dir, "data.yaml")
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    
    logger.info("Starting YOLOv8s training on merged dataset...")
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,  # Match the 640x640 dataset size for optimal accuracy
        batch=batch_size,
        device=device,
        workers=workers,
        amp=amp,
        cache=False,
        project=project_dir,
        name="drone_training",
        save=True,
        verbose=True
    )
    
    # 4. Save best weights to models/best.pt
    import glob
    run_dirs = glob.glob(os.path.join(project_dir, "drone_training*"))
    best_weights_src = None
    if run_dirs:
        # Sort by modification time to get the newest run directory
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        best_weights_src = os.path.join(latest_run_dir, "weights", "best.pt")
        logger.info(f"Detected latest training run directory: {latest_run_dir}")
    else:
        best_weights_src = os.path.join(project_dir, "drone_training", "weights", "best.pt")

    target_weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(target_weights_dir, exist_ok=True)
    target_weights_path = os.path.join(target_weights_dir, "best.pt")
    
    if best_weights_src and os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, target_weights_path)
        logger.info(f"Trained weights successfully copied from {best_weights_src} to: {target_weights_path}")
        
        # Load trained weights to export
        trained_model = YOLO(target_weights_path)
        logger.info("Exporting model to ONNX for optimized edge deployments...")
        try:
            onnx_path = trained_model.export(format="onnx", imgsz=640, simplify=True)
            logger.info(f"Model exported successfully to ONNX: {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
    else:
        logger.error(f"Training finished but best weights were not found at {best_weights_src}")
        
if __name__ == "__main__":
    run_training()
