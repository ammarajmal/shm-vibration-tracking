# shm-vibration-tracking

**Multi-Camera AprilTag 3D Displacement & Vibration Tracking**

This repository contains the official implementation of the research framework for high-fidelity 3D vibration and displacement tracking using consumer cameras. It fuses AprilTag pose estimation with optical flow (KLT) recovery and self-validation metrics to achieve robust offline tracking.

## 🎯 Project Goal
Build a reproducible research codebase and publish **Paper 2**: a novel offline algorithmic framework that recovers 3D motion from multiple views, robust to motion blur and tag dropouts.

## 🏗 System Overview
- **Hardware**: 3x Sony RX10 IV cameras (aiming for 60 FPS), captured via AverMedia capture cards.
- **Capture**: ROS Noetic for recording raw streams to rosbags.
- **Processing**: Heavy computational lifting is done **offline** to avoid realtime constraints (dropped frames, sync errors).

## 📂 Repository Layout
- `ros/`: ROS launch files for recording and playback.
- `offline/`: Core Python library (`shmtrack`) and scripts for processing data.
  - `src/shmtrack/`: Library code (IO, vision, fusion, analysis).
  - `scripts/`: Command-line tools (export, detect, sync, plot).
- `experiments/`: YAML configurations for reproducible runs.
- `data/`: Calibration files and small configs.
- `results/`: Output directory for processed data (ignored by git).

## 🚀 Key Features (Paper 2)
1.  **Robust Tracking**: Fuses AprilTag detection with KLT corner tracking to handle frames where tags are blurred or occluded.
2.  **Self-Validation**: Introduces a **Motion Consistency Index (MCI)** to reject outlier pose updates.
3.  **Offline Sync**: precise time-alignment of multi-camera streams using drift estimation and resampling.
4.  **Signal Analysis**: Frequency-domain stability analysis (FFT) for structural health monitoring (SHM).

## 🛠 Installation

### Prerequisites
- **Ubuntu 20.04** (Recommended)
- **ROS Noetic** (for `rosbag` and `cv_bridge` support)
- **Python 3.8+**

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/ammarajmal/shm-vibration-tracking.git
    cd shm-vibration-tracking
    ```

2.  Create a Python virtual environment and install dependencies:
    ```bash
    cd offline
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    ```

## 🏃 Usage Workflow

1.  **Record Data**: Use ROS launch files to capture camera streams.
    ```bash
    roslaunch shm_ros record_3cams.launch
    ```

2.  **Export & Normalize**: Convert rosbags to PNG frames with normalized timestamps.
    ```bash
    python offline/scripts/bag_export.py --bag <path_to_bag> --out results/<run_name>
    ```

3.  **Run Detection**:
    ```bash
    # Baseline AprilTag only
    python offline/scripts/detect_baseline.py --run-dir results/<run_name> --cam sony_cam1
    ```

4.  **Analyze**: Run fusion and signal analysis scripts (documentation coming soon).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Author
**Ammar Ajmal**
