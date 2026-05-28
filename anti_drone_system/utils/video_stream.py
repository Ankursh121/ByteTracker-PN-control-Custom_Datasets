import cv2
import time
import threading
import logging
import os

logger = logging.getLogger("AntiDroneSystem.VideoStream")

class ThreadedVideoStream:
    def __init__(self, config):
        """
        Initializes the threaded video stream with support for CSI, USB, and RTSP cameras.
        """
        self.config = config
        self.camera_type = config['camera']['type']
        self.camera_source = config['camera']['source']
        self.buffer_size = config['camera']['buffer_size']
        
        self.stream = None
        self.grabbed = False
        self.frame = None
        
        # Thread control
        self.stopped = False
        self.thread = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
        # Performance
        self.read_fps = 0.0

    def _get_csi_pipeline(self, sensor_id=0, capture_w=1280, capture_h=720, display_w=640, display_h=360, framerate=30, flip_method=0):
        """
        Generates a GStreamer pipeline string for NVIDIA Jetson CSI cameras.
        """
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_w}, height=(int){capture_h}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_w}, height=(int){display_h}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        )

    def start(self):
        """
        Starts the background thread to read frames from the video stream.
        """
        # Determine stream source based on camera type
        source = self.camera_source
        
        if self.camera_type == "csi":
            # Attempt to parse sensor ID from source if it's an integer
            try:
                sensor_id = int(source)
            except ValueError:
                sensor_id = 0
            # Construct Jetson GStreamer pipeline
            source = self._get_csi_pipeline(sensor_id=sensor_id)
            logger.info(f"CSI camera requested. Using GStreamer pipeline: {source}")
        else:
            # USB camera source checks
            try:
                source = int(source)
            except ValueError:
                # Video files or RTSP streams remain string paths
                source = str(source)
                
        logger.info(f"Opening video source: {source}...")
        
        # Open OpenCV video stream
        if self.camera_type == "csi":
            self.stream = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            if os.name == 'nt' and (isinstance(source, int) or (isinstance(source, str) and source.isdigit())):
                logger.info(f"Attempting to open camera {source} with CAP_DSHOW backend...")
                self.stream = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
                if self.stream.isOpened():
                    # Warmup/check if DSHOW returns a valid (non-black) frame
                    grabbed, frame = self.stream.read()
                    import numpy as np
                    if not grabbed or frame is None or np.mean(frame) < 1.0:
                        logger.warning("CAP_DSHOW backend returned empty or black frame. Falling back to default MSMF backend...")
                        self.stream.release()
                        self.stream = cv2.VideoCapture(int(source))
            else:
                self.stream = cv2.VideoCapture(source)
            
            if not self.stream.isOpened() and os.name == 'nt':
                # Fallback to default backend if previous attempts failed
                self.stream = cv2.VideoCapture(source)
            
        if not self.stream.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return False

        # Apply default camera property configurations (resolution, FPS, buffer size)
        try:
            if self.camera_type != "csi":
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.stream.set(cv2.CAP_PROP_FPS, 30)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            logger.info(f"Set video capture buffer size to: {self.buffer_size}")
        except Exception as e:
            logger.warning(f"Could not set camera properties: {e}")

        # Try to read the first frame
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            logger.warning("Failed to retrieve initial frame from source. Thread will attempt to recover during warmup.")
            
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("Video stream thread started.")
        return True

    def _update(self):
        """
        Background loop to continuously grab frames from the stream.
        """
        last_time = time.time()
        frame_count = 0
        consecutive_failures = 0
        
        while not self.stopped:
            grabbed, frame = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            
            self.new_frame_event.set()
            
            if consecutive_failures >= 90: # Allow up to 3 seconds of failures (e.g. during camera start/reset)
                logger.warning("Video stream disconnected: 90 consecutive read failures.")
                self.stopped = True
                break
                    
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.read_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
                
            # Short sleep to prevent CPU starvation
            time.sleep(0.001)

    def read(self, wait=False, timeout=None):
        """
        Returns the most recent frame grabbed from the video stream.
        """
        if wait:
            self.new_frame_event.wait(timeout=timeout)
            self.new_frame_event.clear()
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        """
        Stops the thread and releases video capture resources.
        """
        logger.info("Stopping video stream thread...")
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
            
        with self.lock:
            if self.stream:
                self.stream.release()
                self.stream = None
            self.grabbed = False
            self.frame = None
        logger.info("Video stream stopped.")
