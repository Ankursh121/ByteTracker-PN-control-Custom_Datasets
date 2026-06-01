# ORCUS Anti-Drone System: UI Settings Documentation

The **Settings** tab in the ORCUS dashboard allows you to configure the core parameters of the autonomous drone tracking, detection, and flight guidance pipelines. Any changes made here are saved to the backend `settings.yaml` configuration file and require restarting the pipeline to take effect.

Below is a detailed breakdown of every configuration section available in the UI.

---

## 1. YOLO Detection Model
This section configures the primary vision system used to identify the target drone.

* **Model Weights Path**: The file path to the trained YOLOv8 weights (e.g., `models/best.pt`). If you train a new model in the Workshop tab, you can point the system to the new weights here.
* **Confidence Threshold**: The minimum confidence score (0.0 to 1.0) required for the system to consider a detection valid. A higher number reduces false positives (locking onto birds/debris) but might cause the system to drop the lock if the drone is far away.
* **Inference Image Size**: The resolution that the image is resized to before passing it to the neural network (e.g., `640`). Higher sizes increase detection accuracy for small distant drones, but decrease the processing speed (FPS).
* **Hardware Device**: Choose whether to run the neural network on the NVIDIA GPU (`cuda`) for real-time performance, or the Processor (`cpu`) if a GPU is unavailable.

## 2. ByteTrack Association
This configures how the system links frame-by-frame detections together to maintain a persistent ID on the target drone, preventing the camera from randomly swapping targets.

* **Association Track Threshold**: The minimum score required to keep tracking an existing target across multiple frames. 
* **Max Optical Flow Fallback Frames**: When YOLO temporarily loses sight of the target (e.g., due to motion blur or occlusion), the system falls back on a high-speed Lucas-Kanade optical flow tracker. This number dictates the maximum number of consecutive frames the system will use optical flow "dead reckoning" before declaring the target completely lost.

## 3. Pursuit Guidance Law
This section dictates the flight dynamics and how aggressive the interceptor drone acts when tracking the target.

* **Proportional Nav Gain (N)**: The navigation constant used in Proportional Navigation (PN). A higher number (e.g., `3.0` to `5.0`) makes the drone turn much more aggressively to cut off the target's trajectory. A lower number makes the pursuit smoother but slower.
* **Desired Follow Standoff (m)**: When using the "Follow Target" mode, this is the exact distance (in meters) the interceptor will attempt to maintain behind the target drone.
* **Max XY Speed Command (m/s)**: The hard speed limit imposed on the interceptor drone. The system will clamp all generated velocity commands to ensure the drone doesn't exceed this maximum horizontal speed.
* **Target Loss Timeout (s)**: The number of seconds the system will wait after completely losing visual lock before automatically stopping the drone and triggering safety protocols (like returning to launch or hovering).

## 4. MAVLink Connection (SpeedyBee)
This section tells the backend how to talk to the physical flight controller.

* **Link Type**: 
  * *Serial Port*: Used for direct hardware connections via USB or UART telemetry radios.
  * *UDP / TCP*: Used primarily for communicating with software simulators (SITL).
* **Serial COM Port / Dev**: The specific port your SpeedyBee flight controller is connected to (e.g., `COM9` on Windows, or `/dev/ttyUSB0` on Linux). 
* **UDP Target Address & UDP Port**: The network IP and Port used to route MAVLink packets when running in a simulated environment.

## 5. Video Acquisition Source
This configures where the system pulls its live video feed from.

* **System Operating Mode**:
  * *SITL Simulation*: Connects to a virtual environment for safe software-in-the-loop testing.
  * *Local Webcam*: Uses a USB webcam connected directly to your computer.
  * *Hardware Camera Deployment*: Optimizes settings for the physical onboard companion computer (like a Raspberry Pi or Jetson).
* **Camera Index / RTSP URL / File path**:
  * If the mode is set to **Local Webcam**, this transforms into a dropdown menu where you can explicitly select which connected webcam to use (e.g., "Integrated Camera" vs "External USB Camera").
  * You can also select "Custom URL / File" to manually enter an RTSP network stream link or the path to a recorded `.mp4` video for testing.

---

### Saving Your Configuration
After making changes, click the **Save Settings & Restart Pipeline** button at the bottom. This will write your changes to the disk, safely spin down the active tracking processes, and reboot the vision and control loops with your new parameters.




1. The New Hybrid Logic (Backend)
I modified web_server.py to add true "Hybrid Follow" capability. When the system is in FOLLOW mode, it now dynamically checks the distance to the target.

If the target is far away, it automatically engages Direct Pursuit to aggressively close the gap at high speed.
When it gets close (approaching your Desired Follow Distance), it smoothly transitions into Follow Target mode to station-keep and hover exactly behind it.
2. The DESTROY Logic (Backend)
When the system is in DESTROY mode, it immediately locks in the Proportional Navigation (PN) algorithm to calculate an intercept trajectory and flies a direct collision course.

3. The Streamlined UI
I completely stripped out the old LAW dropdown bar and the Active Follow slider. In their place, directly below your video feed, you now have two large engagement buttons:

🔵 [ FOLLOW ]
🔴 [ DESTROY ]
How they work: These act as master toggle buttons. If you click FOLLOW, it automatically activates the follow tracking and shifts the backend into the Hybrid logic. If you click DESTROY, it immediately shifts to the Proportional Navigation logic. If you click the currently active button a second time, it turns off following and puts the drone back into standby/hover mode.