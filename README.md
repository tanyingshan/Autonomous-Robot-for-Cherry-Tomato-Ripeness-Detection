# Autonomous Robot for Cherry Tomato Ripeness Detection

##  Project Overview
This ROS 2 package controls an autonomous robotic arm to identify, approach, and harvest ripe cherry tomatoes. It integrates **YOLOv8** for real-time detection, **RGB-D Perception** for depth estimation, and **MoveIt** for motion planning. A user-friendly **Streamlit Dashboard** is included for live monitoring.



### Key Features
* **Dual-Stage Perception:** YOLOv8 for ripeness classification + Instance Segmentation for precise stem cutting points.
* **Robust Depth Search:** Custom "Spiral Search" algorithm to recover valid 3D coordinates from noisy depth data.
* **Interactive Dashboard:** Web-based UI (Streamlit) displaying live camera feed, mask overlays, and real-time telemetry.
* **Autonomous State Machine:** Full cycle control (Detect → Approach → Grip → Cut → Reset).

---

##  Prerequisites

Ensure you have **Ubuntu 22.04** and **ROS 2 (Humble or Iron)** installed.

### 1. System Dependencies
```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-moveit

```

### 2. Python Libraries

```bash
pip install ultralytics streamlit opencv-python numpy

```

### 3. Hardware Drivers

Ensure you have the drivers for your specific hardware installed in your workspace:

* **Camera:** `orbbec_camera` (Gemini RGB-D)
* **Robot:** `cocoabot_bringup`

---

##  Installation

1. **Create a Workspace:**
```bash
mkdir -p ~/tomato_harvest_ws/src
cd ~/tomato_harvest_ws/src

```


2. **Add the Package:**
Copy the `tomato_perception` folder into the `src` directory.
3. **Build the Workspace:**
```bash
cd ~/tomato_harvest_ws
colcon build --packages-select tomato_perception
source install/setup.bash

```



---

##  Usage Guide

### Terminal 1: Camera Driver (Run First)

Launch the RGB-D camera driver separately to ensure the video stream is stable.

```bash
ros2 launch orbbec_camera gemini.launch.xml

```

### Terminal 2: Cherry Tomato Ripeness Detection System

Starts the Robot Arm, Perception AI, Harvest Controller, and User Interface simultaneously.

```bash
ros2 launch tomato_perception tomato_harvest_system.launch.py

```

### 3. Open Dashboard

Once the system starts, open your web browser and go to:
[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
