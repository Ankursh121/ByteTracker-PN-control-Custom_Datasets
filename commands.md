# Command Reference Guide

This reference guide contains all the commands needed to set up, train, evaluate, simulate, and deploy the **Real-Time Autonomous Drone Pursuit & Anti-Drone Tracking System**.

---

## ⚙️ 1. Environment Setup

Run these commands from the root directory (`d:\Bytetrack`) to initialize your environment and install dependencies.

### Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
```

### Install Required Libraries
```powershell
pip install -r anti_drone_system/requirements.txt
```

---

## 🚀 2. Vision Guidance & Pursuit Pipeline (`anti_drone_system`)

Navigate to the `anti_drone_system` directory to execute these commands, or run them from the root by prefixing with the directory name.

```powershell
# Change directory to the pipeline workspace
cd d:\Bytetrack\anti_drone_system
```

### Run in Virtual Simulation Mode (Offline SITL)
Test the vision tracking, Kalman filter state prediction, and proportional navigation (PN) guidance law using a virtual target:
```powershell
python main.py --mode simulation
```

### Run with Local Webcam Feed
Run target detection and visual tracking using a local webcam:
```powershell
python main.py --mode webcam --source 0
```

### Run in Hardware Deployment (Jetson Nano / companion computer connected to SpeedyBee FC)
Execute the active guidance pipeline on the drone, receiving video from a USB/CSI camera and sending real-time MAVLink navigation velocities to the flight controller:
```powershell
python main.py --mode hardware --source "/dev/video0"
```

### CLI Command Options Cheat Sheet
Modify parameters on-the-fly using these command arguments:
| Flag | Options / Format | Explanation |
| :--- | :--- | :--- |
| `--mode` | `simulation`, `webcam`, `hardware`, `sitl` | Selects system running mode. |
| `--model` | Path to `.pt` weights (e.g. `models/best.pt`) | Overrides the default YOLO model weight file. |
| `--source` | Camera index, RTSP stream URL, or video path | Overrides the camera input device source. |
| `--no-gui` | *(Flag)* | Disables the OpenCV graphical interface (recommended for headless Jetson). |

*Example using overrides:*
```powershell
python main.py --mode hardware --source 1 --model models/best.pt --no-gui
```

### 🎮 GUI Window Controls (HUD Keybinds)
When running the OpenCV visualizer window, use these keyboard hotkeys to interact with the flight controller and guidance law:
* **`q`** : Quit system safely (sends zero-velocity hover commands before disconnecting).
* **`p`** : Pause/resume pipeline processing.
* **`t`** : **Toggle Autonomous Guidance** (Enable/Disable sending pursuit velocities).
* **`f`** : **Toggle Follow Mode Safety Switch** (Enable/Disable active target following).
* **`c`** : Cycle guidance law (**Follow Target** $\Leftrightarrow$ **Proportional Navigation** $\Leftrightarrow$ **Direct Pursuit**).
* **`d`** : Toggle debug visualization overlay (3D predicted positions, velocity vectors).
* **`a`** / **`s`** : Send **ARM** / **DISARM** flight commands.
* **`o`** : Send **TAKEOFF** command (commands climb to a default 3-meter altitude).
* **`l`** / **`r`** : Send **LAND** / **Return-To-Launch (RTL)** modes.
* **`e`** : **EMERGENCY STOP** (instantly halts auto-guidance and commands hover).

---

## 🏋️ 3. Dataset Preprocessing & Model Training (`coco json drone detection`)

Navigate to the dataset workspace to prepare datasets and train custom YOLO models.

```powershell
# Change directory to dataset workspace
cd "d:\Bytetrack\coco json drone detection"
```

### Step 1: Preprocess, Resize, and Merge Custom Datasets
Converts original COCO JSON datasets to single-class (class ID 0: Drone) YOLO format, cleans duplicates, and resizes raw images to `640x640`:
```powershell
python restructure_custom_dataset.py
```

### Step 2: Apply Offline Dataset Augmentations (Optional)
Applies geometric transformations, motion blurs, sensor noise, and weather simulations to expand dataset diversity:
```powershell
python augment_dataset.py
```

### Step 3: Run CLAHE & SAHI-style Overlapping Tiling (Recommended)
Performs local contrast enhancements and slices high-resolution images containing tiny targets into overlapping tiles, forcing the network to detect small/distant drones:
```powershell
python enhance_dataset.py
```

### Step 4: Auto-Label / Pseudo-Label Unannotated Categories
Uses a temporarily trained model to generate labels for new unannotated drone images:
```powershell
python auto_annotate.py
```
> [!NOTE]
> Ensure you copy the generated files from `dataset/auto_labeled/` into `dataset/images/train/` and `dataset/labels/train/` to include them in the final training dataset.

### Step 5: Start Model Training
Runs hardware-aware optimizations to automatically configure worker threads, batch sizes, and train a YOLO model:

#### Option A: Strict Drone-Only Model (Recommended to prevent false positives)
Starts the strict training run featuring cosine learning rate scheduler, advanced small-target augmentation multipliers, and early stopping patience checks:
```powershell
python train_strict_model.py
```

#### Option B: Standard Training Run
```powershell
python train_model.py
```

### Step 6: Validate & Evaluate Test Split Performance
Generates detailed False Positive analyses, listing performance precision, recall, and identifying potential false locks (like birds or background noise):
```powershell
python evaluate.py
```

---

## 🔬 4. Advanced Hardware-Aware Training Commands (CLI Direct)

Use these raw YOLO CLI commands to train directly if you want manual control over hyperparameters:

### High-Performance NVIDIA GPU (VRAM $\ge$ 10GB)
```powershell
yolo detect train data="dataset_merged/data.yaml" model=yolov8s.pt epochs=100 imgsz=640 batch=32 device=0 workers=8 amp=True project=runs name=rtx_opt
```

### Low VRAM GPU (VRAM $\le$ 6GB)
```powershell
yolo detect train data="dataset_merged/data.yaml" model=yolov8s.pt epochs=100 imgsz=640 batch=8 device=0 workers=2 amp=True project=runs name=low_vram_opt
```

### CPU Fallback Mode
```powershell
yolo detect train data="dataset_merged/data.yaml" model=yolov8s.pt epochs=20 imgsz=640 batch=2 device=cpu workers=0 project=runs name=cpu_fallback
```

---

## 🛡️ 5. Specialized Anti-Drone Strict Monitor

This engine implements strict geometrical shape validation (aspect ratio checks, area checks), Region of Interest filtering (ignores floor and ceiling corners), and static background suppression combined with a hybrid Lucas-Kanade optical flow tracker.

```powershell
# Change directory to the dataset/strict engine workspace
cd "d:\Bytetrack\coco json drone detection"

# Launch the interactive detector CLI
python strict_detector.py
```

### Stream Interactive Keybinds
* **`c`** / **`v`** : Increase / decrease detection confidence threshold dynamically (+/- 0.05).
* **`d`** : Toggle debug graphics (visualize optical flow keypoints and rejected boxes).
* **`q`** / **`ESC`** : Exit the monitoring stream.
