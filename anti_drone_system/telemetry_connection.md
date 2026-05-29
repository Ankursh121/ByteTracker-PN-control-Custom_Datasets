# 🔌 MAVLink Telemetry Connection Guide

Complete reference for connecting the ORCUS Anti-Drone System to your flight controller via MAVLink — covering SITL simulation, USB serial (SpeedyBee), and companion computer UART deployment.

---

## 📋 Prerequisites

Install the Python MAVLink library if not already present:
```powershell
pip install pymavlink
```

---

## Mode 1 — SITL Simulation (No Hardware Required)

Use this mode for testing the full pipeline on your PC with a virtual drone.

### `settings.yaml` Configuration
```yaml
mavlink:
  connection_type: udp
  udp_address: 127.0.0.1
  udp_port: 14550
system:
  simulation_mode: true
```

### Steps

1. **Install Mission Planner** (Windows) or ArduPilot SITL:
   - Mission Planner: https://ardupilot.org/planner/docs/mission-planner-installation.html
   - ArduPilot SITL: https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html

2. **Launch SITL** in Mission Planner:
   - Go to **Simulation** tab
   - Select **Multirotor** → Click **Start**
   - SITL automatically outputs MAVLink on `UDP port 14550`

3. **Start the backend:**
   ```powershell
   python anti_drone_system/web_server.py
   ```

4. **Verify in UI:**
   - Header badge shows: `FC Link: GUIDED`
   - Live Telemetry Stats → `STATUS: LINK OK`

> **Note:** In simulation mode, all flight control commands (Arm, Takeoff, Land, etc.) are processed by the SITL virtual vehicle, not real hardware.

---

## Mode 2 — SpeedyBee FC via USB Serial (Hardware)

Use this mode when your SpeedyBee F405/F7 flight controller is directly connected to your development PC via USB.

### `settings.yaml` Configuration
```yaml
mavlink:
  connection_type: serial
  serial_port: COM3        # ← update to your actual COM port
  baudrate: 921600
system:
  simulation_mode: false
```

### Steps

#### 1. Find Your COM Port
- Open **Device Manager** on Windows
- Expand **Ports (COM & LPT)**
- Look for **USB Serial Device** or **SpeedyBee** → note the port (e.g. `COM3`, `COM6`)
- Update `serial_port` in `settings.yaml`

#### 2. Configure SpeedyBee in Mission Planner (ArduPilot)
- Open **Mission Planner** → connect to your FC via USB
- Go to **Config** → **Full Parameter List**
- Find the UART port you are using (e.g. `SERIAL2` for UART2):
  - Set `SERIAL2_PROTOCOL` = `2` (MAVLink 2)
  - Set `SERIAL2_BAUD` = `921` (for 921600 baud)
- Click **Write Params** and **Reboot** the flight controller

#### 3. Set Flight Mode to GUIDED
- In Mission Planner, ensure the flight mode is set to **GUIDED** for autonomous control
- Alternatively the backend will request GUIDED mode automatically on connection

#### 4. Start the Backend
```powershell
python anti_drone_system/web_server.py
```

#### 5. Verify Connection
- UI header: `FC Link: GUIDED`
- Dashboard → Live Telemetry Stats:
  - `STATUS: LINK OK`
  - `ARMED: DISARMED`
  - Battery voltage shows real value (e.g. `16.4 V`)

---

## Mode 3 — Companion Computer via UART (Field Deployment)

Use this mode when a Raspberry Pi, Jetson Nano, or similar SBC is mounted on the drone and connected directly to the SpeedyBee's UART pins.

### `settings.yaml` Configuration
```yaml
mavlink:
  connection_type: serial
  serial_port: /dev/ttyAMA0   # Pi hardware UART
  # or /dev/ttyUSB0           # if using a USB-to-serial adapter
  baudrate: 921600
system:
  simulation_mode: false
```

### Wiring Diagram

```
SpeedyBee FC                Raspberry Pi / Jetson
─────────────               ──────────────────────
UART2 TX  ──────────────►  GPIO15 / Pin 10  (RX)
UART2 RX  ◄──────────────  GPIO14 / Pin 8   (TX)
GND       ───────────────  GND              (Pin 6)
```

> ⚠️ **NEVER connect 5V/3.3V power lines between FC and Pi over UART — GND only.**

### Steps

#### 1. Enable UART on Raspberry Pi
```bash
sudo raspi-config
# Interface Options → Serial Port
#   → Shell over serial?  → No
#   → Hardware serial?    → Yes
sudo reboot
```

#### 2. Disable Serial Console (prevents conflict)
Edit `/boot/cmdline.txt` and remove:
```
console=serial0,115200
```

#### 3. Grant Serial Port Permissions
```bash
sudo usermod -aG dialout $USER
# Log out and back in, or run:
newgrp dialout
```

#### 4. Verify Device is Available
```bash
ls /dev/ttyAMA0     # hardware UART
ls /dev/ttyUSB0     # USB-serial adapter
```

#### 5. Configure SpeedyBee UART in ArduPilot
In **Mission Planner** → **Config** → **Full Parameter List**:
- Set `SERIALx_PROTOCOL` = `2` (MAVLink 2, where x is your UART number)
- Set `SERIALx_BAUD` = `921` (921600)
- Click **Write Params** and **Reboot**

#### 6. Start the Backend
```bash
python anti_drone_system/web_server.py
```

---

## ✅ Verifying Connection in the Web UI

Once connected, check these indicators:

| Location | Indicator | Meaning |
|---|---|---|
| **Header badge** | `FC Link: GUIDED` 🟢 | Connected, ready for commands |
| **Header badge** | `FC Link: Disconnected` 🔴 | Check port/baud/cable |
| **Telemetry Stats** | `STATUS: LINK OK` | MAVLink heartbeat active |
| **Telemetry Stats** | `ARMED: DISARMED` | FC responding correctly |
| **Telemetry Stats** | Battery voltage > 0 V | Real telemetry flowing |

---

## 🔧 Changing Connection Mode Live (via UI)

You can switch the connection type without editing `settings.yaml` manually:

1. Open `http://localhost:5173`
2. Go to **Settings** tab
3. Scroll to **MAVLink Connection (SpeedyBee)** section
4. Change **Link Type** → Select `Serial`, `UDP`, or `TCP`
5. Update the port/address fields
6. Click **Save Settings & Restart Pipeline**

The backend will reload the config and attempt the new connection automatically.

---

## 🛠️ Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `FC Link: Disconnected` | Wrong COM port | Check Device Manager, update `serial_port` |
| `FC Link: Disconnected` | Wrong baud rate | Match `baudrate` to ArduPilot `SERIALx_BAUD` setting |
| `FC Link: Disconnected` (UDP) | SITL not running | Start Mission Planner SITL first |
| `LINK OK` but no telemetry values | MAVLink stream rates not set | Connect via Mission Planner and request streams (SRx_ params) |
| `Permission denied /dev/ttyAMA0` | Missing dialout group | `sudo usermod -aG dialout $USER` |
| Serial port in use | Mission Planner still connected | Click Disconnect in Mission Planner |
| `WinError 10038` (UDP) | Socket closed/no SITL | Start SITL before the backend |

---

## ⚙️ Connection String Reference

| Type | Connection String Format | Example |
|---|---|---|
| Serial | `COMx` / `/dev/ttyXXX` | `COM3` / `/dev/ttyAMA0` |
| UDP (listen) | `udpin:HOST:PORT` | `udpin:127.0.0.1:14550` |
| TCP (connect) | `tcp:HOST:PORT` | `tcp:127.0.0.1:5762` |

---

## 📡 Supported Baud Rates

| Connection | Recommended Baud |
|---|---|
| SpeedyBee USB (testing) | `115200` |
| SpeedyBee UART (companion) | `921600` |
| RFD900 Telemetry Radio | `57600` |
| SiK Radio (3DR/Holybro) | `57600` |
