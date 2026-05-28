import time
import threading
import logging
from pymavlink import mavutil

logger = logging.getLogger("AntiDroneSystem.MAVLink")

class MAVLinkConnectionManager:
    def __init__(self, config):
        """
        Initializes the MAVLink Connection Manager with support for SpeedyBee and ArduPilot.
        """
        self.config = config
        self.connection_type = config['mavlink']['connection_type']
        self.serial_port = config['mavlink']['serial_port']
        self.baudrate = config['mavlink']['baudrate']
        self.udp_address = config['mavlink']['udp_address']
        self.udp_port = config['mavlink']['udp_port']
        self.tcp_address = config['mavlink']['tcp_address']
        self.tcp_port = config['mavlink']['tcp_port']
        
        self.heartbeat_rate = config['mavlink']['heartbeat_rate']
        self.safety_timeout = config['mavlink']['safety_timeout']
        self.reconnect_interval = config['mavlink']['reconnect_interval']
        self.command_rate_hz = config['mavlink']['command_rate_hz']
        
        self.system_id = config['mavlink']['system_id']
        self.target_system_id = config['mavlink']['target_system_id']
        self.target_component_id = config['mavlink']['target_component_id']
        self.simulation_mode = config['system']['simulation_mode']
        self.enable_msg_logging = config['mavlink']['enable_msg_logging']
        
        # State variables
        self.master = None
        self.is_connected = False
        self.is_armed = False
        self.current_mode = "UNKNOWN"
        self.battery_voltage = 0.0
        self.altitude = 0.0
        self.gps_lock = "NO_GPS"
        self.last_heartbeat_received = 0
        
        # Attitude and Position Telemetry
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0
        
        # Rate limiting state
        self.last_command_time = 0
        self.command_min_interval = 1.0 / self.command_rate_hz
        
        # Thread controls
        self.running = False
        self.connection_thread = None
        self.heartbeat_thread = None
        self.rx_thread = None
        self.lock = threading.Lock()
        
        # Track last failsafe execution time
        self.last_failsafe_trigger = 0
        self.last_sim_time = time.time()

    def connect(self):
        """
        Starts the background connection thread.
        """
        if self.simulation_mode:
            logger.info("MAVLink running in SIMULATION MODE. Hardware control disabled.")
            self.is_connected = True
            self.current_mode = "GUIDED"
            self.last_sim_time = time.time()
            return True

        self.running = True
        self.connection_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.connection_thread.start()
        logger.info("MAVLink background connection manager started.")
        return True

    def _connection_loop(self):
        """
        Background loop that attempts to connect and handle reconnections.
        """
        while self.running:
            if not self.is_connected:
                connection_string = ""
                baud = None

                # Build connection URI based on configuration
                if self.connection_type == "serial":
                    connection_string = self.serial_port
                    baud = self.baudrate
                    logger.info(f"Connecting to SpeedyBee UART on {connection_string} at {baud} baud...")
                elif self.connection_type == "udp":
                    connection_string = f"udpin:{self.udp_address}:{self.udp_port}"
                    logger.info(f"Listening for UDP MAVLink packets on {connection_string}...")
                elif self.connection_type == "tcp":
                    connection_string = f"tcp:{self.tcp_address}:{self.tcp_port}"
                    logger.info(f"Connecting to TCP MAVLink endpoint at {connection_string}...")

                try:
                    with self.lock:
                        if self.master:
                            self.master.close()
                            self.master = None
                        
                        if self.connection_type == "serial":
                            self.master = mavutil.mavlink_connection(
                                connection_string,
                                baud=baud,
                                source_system=self.system_id
                            )
                        else:
                            self.master = mavutil.mavlink_connection(
                                connection_string,
                                source_system=self.system_id
                            )

                    # Start telemetry and heartbeat threads if not already running
                    if not self.rx_thread or not self.rx_thread.is_alive():
                        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
                        self.rx_thread.start()

                    if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
                        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                        self.heartbeat_thread.start()

                    # Wait a bit for the first heartbeat
                    time.sleep(self.reconnect_interval)

                except Exception as e:
                    logger.error(f"MAVLink setup failed: {e}. Retrying in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
            else:
                time.sleep(1.0)

    def disconnect(self):
        """
        Shuts down threads and closes connection.
        """
        logger.info("Disconnecting MAVLink manager...")
        self.running = False
        
        if self.connection_thread:
            self.connection_thread.join(timeout=1.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)

        with self.lock:
            if self.master:
                self.master.close()
                self.master = None
            self.is_connected = False
            self.is_armed = False
            logger.info("MAVLink disconnected.")

    def set_mode(self, mode_name):
        """
        Changes ArduPilot mode (e.g. 'GUIDED', 'RTL', 'LAND', 'LOITER').
        """
        if self.simulation_mode:
            logger.info(f"[Sim] Mode switch requested: {mode_name}")
            self.current_mode = mode_name
            return True

        if not self.master or not self.is_connected:
            logger.error(f"Cannot set mode {mode_name}: Not connected.")
            return False

        logger.info(f"Sending request to switch to mode: {mode_name}")
        
        mode_id = self.master.mode_mapping().get(mode_name)
        if mode_id is None:
            logger.error(f"Flight mode {mode_name} not available in mode mapping: {list(self.master.mode_mapping().keys())}")
            return False

        try:
            self.master.set_mode(mode_id)
            return True
        except Exception as e:
            logger.error(f"Set Mode command failed: {e}")
            return False

    def arm(self):
        """
        Sends standard command to arm the vehicle.
        """
        if self.simulation_mode:
            logger.info("[Sim] Arming vehicle.")
            self.is_armed = True
            return True

        if not self.master or not self.is_connected:
            logger.error("Arming failed: Not connected.")
            return False

        logger.info("Sending Arm command (MAV_CMD_COMPONENT_ARM_DISARM)...")
        try:
            self.master.mav.command_long_send(
                self.target_system_id,
                self.target_component_id,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                1,  # 1 = Arm, 0 = Disarm
                0, 0, 0, 0, 0, 0  # Unused params
            )
            return True
        except Exception as e:
            logger.error(f"Arming failed: {e}")
            return False

    def disarm(self):
        """
        Sends standard command to disarm the vehicle.
        """
        if self.simulation_mode:
            logger.info("[Sim] Disarming vehicle.")
            self.is_armed = False
            return True

        if not self.master or not self.is_connected:
            logger.error("Disarming failed: Not connected.")
            return False

        logger.info("Sending Disarm command...")
        try:
            self.master.mav.command_long_send(
                self.target_system_id,
                self.target_component_id,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,  # 0 = Disarm
                0, 0, 0, 0, 0, 0
            )
            return True
        except Exception as e:
            logger.error(f"Disarming failed: {e}")
            return False

    def takeoff(self, altitude_m=3.0):
        """
        Commands the vehicle to takeoff to a specific target altitude (m).
        """
        if self.simulation_mode:
            logger.info(f"[Sim] Taking off to {altitude_m}m.")
            self.altitude = altitude_m
            self.local_z = -altitude_m
            return True

        if not self.master or not self.is_connected:
            logger.error("Takeoff failed: Not connected.")
            return False

        if self.current_mode != "GUIDED":
            logger.warning("Takeoff command requires GUIDED flight mode. Attempting mode switch...")
            self.set_mode("GUIDED")
            time.sleep(0.5)

        logger.info(f"Sending Takeoff command (MAV_CMD_NAV_TAKEOFF) for {altitude_m}m...")
        try:
            self.master.mav.command_long_send(
                self.target_system_id,
                self.target_component_id,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0,
                altitude_m
            )
            return True
        except Exception as e:
            logger.error(f"Takeoff command failed: {e}")
            return False

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """
        Sends local body frame velocity and yaw rate commands.
        Enforces command rate limiting and checks armed state.
        """
        now = time.time()
        
        if now - self.last_command_time < self.command_min_interval:
            return
            
        self.last_command_time = now
        
        if self.simulation_mode:
            now_sim = time.time()
            dt = now_sim - self.last_sim_time
            self.last_sim_time = now_sim
            
            if dt > 0.5:
                dt = 0.05
                
            if self.is_armed:
                if self.current_mode == "LAND":
                    self.local_z += 1.0 * dt
                    self.altitude = -self.local_z
                    if self.local_z >= 0.0:
                        self.local_z = 0.0
                        self.altitude = 0.0
                        self.is_armed = False
                elif self.current_mode == "RTL":
                    dx = -self.local_x
                    dy = -self.local_y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0.5:
                        self.local_x += (dx / dist) * 4.0 * dt
                        self.local_y += (dy / dist) * 4.0 * dt
                        self.yaw = np.arctan2(dy, dx)
                    else:
                        self.local_x = 0.0
                        self.local_y = 0.0
                        self.current_mode = "LAND"
                else:
                    self.yaw += yaw_rate * dt
                    self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
                    
                    cos_yaw = np.cos(self.yaw)
                    sin_yaw = np.sin(self.yaw)
                    v_north = vx * cos_yaw - vy * sin_yaw
                    v_east  = vx * sin_yaw + vy * cos_yaw
                    v_down  = -vz
                    
                    self.local_x += v_north * dt
                    self.local_y += v_east * dt
                    self.local_z += v_down * dt
                    self.altitude = -self.local_z
            return

        if not self.master or not self.is_connected:
            return

        if not self.is_armed:
            logger.warning("MAVLink block: Denied velocity command because vehicle is DISARMED.")
            return

        if self.current_mode != "GUIDED":
            logger.warning(f"MAVLink block: Denied command because mode is {self.current_mode} (GUIDED required).")
            return

        try:
            # Command local target velocities in body-NED frame:
            type_mask = 0x07C7
            
            self.master.mav.set_position_target_local_ned_send(
                0,                          # time_boot_ms (ignored)
                self.target_system_id,       # target_system
                self.target_component_id,    # target_component
                8,                          # coordinate_frame: MAV_FRAME_BODY_NED
                type_mask,                  # type_mask
                0, 0, 0,                    # positions (ignored)
                vx, vy, -vz,                # velocities (m/s) (vz negated for up command)
                0, 0, 0,                    # accelerations (ignored)
                0,                          # yaw angle (ignored)
                yaw_rate                    # yaw rate (rad/s)
            )
        except Exception as e:
            logger.error(f"Error dispatching velocity command packet: {e}")

    def _heartbeat_loop(self):
        """
        Sends background heartbeats to keep the MAVLink connection alive.
        """
        while self.running:
            if self.is_connected and self.master:
                try:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                except Exception as e:
                    logger.debug(f"Failed to transmit heartbeat: {e}")
            time.sleep(self.heartbeat_rate)

    def _rx_loop(self):
        """
        Continuously listens to incoming telemetry messages from ArduPilot.
        """
        self.last_heartbeat_received = time.time()
        
        while self.running:
            try:
                if not self.master:
                    time.sleep(0.1)
                    continue

                msg = self.master.recv_match(blocking=True, timeout=0.1)
                
                if msg is None:
                    self._check_failsafe()
                    continue

                if self.enable_msg_logging:
                    logger.debug(f"MAVLink Rx Message: {msg.get_type()}")

                msg_type = msg.get_type()

                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat_received = time.time()
                    if not self.is_connected:
                        logger.info("MAVLink Heartbeat detected! Link established.")
                    
                    with self.lock:
                        self.is_connected = True
                        
                        mode_id = msg.custom_mode
                        mode_map = self.master.mode_mapping()
                        self.current_mode = "UNKNOWN"
                        for name, val in mode_map.items():
                            if val == mode_id:
                                self.current_mode = name
                                break
                                
                        self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                elif msg_type == 'SYS_STATUS':
                    with self.lock:
                        self.battery_voltage = msg.voltage_battery / 1000.0

                elif msg_type == 'VFR_HUD':
                    with self.lock:
                        self.altitude = msg.alt

                elif msg_type == 'ATTITUDE':
                    with self.lock:
                        self.roll = msg.roll
                        self.pitch = msg.pitch
                        self.yaw = msg.yaw

                elif msg_type == 'LOCAL_POSITION_NED':
                    with self.lock:
                        self.local_x = msg.x
                        self.local_y = msg.y
                        self.local_z = msg.z

                elif msg_type == 'GPS_RAW_INT':
                    with self.lock:
                        fix = msg.fix_type
                        if fix == 0 or fix == 1:
                            self.gps_lock = "NO_GPS"
                        elif fix == 2:
                            self.gps_lock = "2D_LOCK"
                        elif fix >= 3:
                            self.gps_lock = f"3D_LOCK ({msg.satellites_visible} Sats)"

            except Exception as e:
                logger.error(f"Error reading MAVLink telemetry channel: {e}")
                with self.lock:
                    self.is_connected = False
                time.sleep(0.5)

            self._check_failsafe()

    def _check_failsafe(self):
        """
        Checks if connection has timed out or if target loss timeout occurred.
        """
        now = time.time()

        if self.is_connected and not self.simulation_mode:
            if now - self.last_heartbeat_received > 5.0:
                logger.warning("Heartbeat telemetry packet stream lost from autopilot!")
                with self.lock:
                    self.is_connected = False
                    self.is_armed = False
                    self.current_mode = "UNKNOWN"
                return

        if self.is_connected and self.is_armed:
            if self.last_command_time > 0 and (now - self.last_command_time > self.safety_timeout):
                if now - self.last_failsafe_trigger > 2.0:
                    logger.warning("Pursuit safety timeout: No command updates. Commanding safe hover.")
                    try:
                        self.master.mav.set_position_target_local_ned_send(
                            0, self.target_system_id, self.target_component_id,
                            8, 0x07C7, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0
                        )
                    except Exception:
                        pass
                    self.last_failsafe_trigger = now
