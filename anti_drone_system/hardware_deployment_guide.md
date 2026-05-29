# 🚁 ORCUS Anti-Drone System: Hardware Deployment Guide

This guide covers how to physically deploy the ORCUS Anti-Drone vision pipeline onto a companion computer (like a Raspberry Pi 4, Jetson Nano, or Orange Pi) and hardwire it directly to your SpeedyBee Flight Controller for autonomous field operations.

---

## 1. Hardware Wiring (UART)

You must wire a free UART port on the SpeedyBee FC directly to the GPIO serial pins on your companion computer. 

> ⚠️ **CRITICAL WARNING:** NEVER connect the 5V or 3.3V power pins between the flight controller and the companion computer via the UART header. It will fry your boards. Connect **ONLY** TX, RX, and GND. 

### Wiring Diagram
```text
SpeedyBee FC                Companion Computer (e.g., Raspberry Pi)
─────────────               ───────────────────────────────────────
UARTx TX  ──────────────►   GPIO RX  (Pin 10 on Pi)
UARTx RX  ◄──────────────   GPIO TX  (Pin 8 on Pi)
GND       ───────────────   GND      (Pin 6 on Pi)
```
*(Note: TX always goes to RX, and RX always goes to TX!)*

---

## 2. Prepare the Companion Computer OS

By default, Linux (especially Raspberry Pi OS) uses the hardware serial pins to output a terminal console. We must disable this so our Python backend can use the pins to send MAVLink commands instead.

### On Raspberry Pi:
1. Open a terminal and run the configuration tool:
   ```bash
   sudo raspi-config
   ```
2. Navigate to **Interface Options** → **Serial Port**.
3. Answer the prompts exactly like this:
   - *Would you like a login shell to be accessible over serial?* ➔ **No**
   - *Would you like the serial port hardware to be enabled?* ➔ **Yes**
4. Grant your user permission to read/write to the serial port:
   ```bash
   sudo usermod -aG dialout $USER
   ```
5. **Reboot the companion computer.**

---

## 3. Configure ArduPilot for High-Speed MAVLink

For the anti-drone system to track effectively, the flight controller needs to receive commands and send telemetry back at very high speeds.

1. Connect the flight controller to **Mission Planner** via USB (one last time).
2. Go to **Config** → **Full Parameter List**.
3. Assuming you wired the companion computer to **UART2** (which is `SERIAL2` in ArduPilot), update the following:
   - **`SERIAL2_PROTOCOL`** = **`2`** *(Sets the port to use MAVLink 2)*
   - **`SERIAL2_BAUD`** = **`921`** *(Sets the baud rate to 921600 for high-speed edge computing)*
4. Set the auto-streaming rates so the drone constantly broadcasts its state:
   - **`SR2_EXTRA1`** = **`10`**
   - **`SR2_EXTRA2`** = **`10`**
   - **`SR2_POSITION`** = **`10`**
   - **`SR2_EXT_STAT`** = **`2`**
5. Click **Write Params** and **Reboot** the flight controller.

---

## 4. Configure ORCUS `settings.yaml`

Now configure the backend on the companion computer to look for the Linux hardware serial port instead of a Windows COM port.

Edit `anti_drone_system/configs/settings.yaml`:
```yaml
mavlink:
  connection_type: serial
  serial_port: /dev/ttyAMA0    # Standard hardware UART on Raspberry Pi
  # serial_port: /dev/ttyUSB0  # Use this instead IF using a USB-to-Serial adapter!
  baudrate: 921600             # Must match the SERIALx_BAUD you set in ArduPilot!
```

---

## 5. Launch the System

With the drone powered by its LiPo battery and the companion computer running, start the backend on the companion computer:

```bash
cd ~/Bytetrack
source venv/bin/activate  # (if using a virtual environment)
python anti_drone_system/web_server.py
```

Open a web browser on your phone or laptop (connected to the same WiFi network as the companion computer) and go to:
`http://<COMPANION_COMPUTER_IP>:5173`

You should instantly see the camera feed and the MAVLink telemetry reporting `STATUS: LINK OK`. You are now ready for autonomous field testing!
