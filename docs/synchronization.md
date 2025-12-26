# Offline Synchronization

Since our cameras are not hardware-triggered, they drift relative to each other over long captures. Precise varying time alignment is critical for multi-view fusion.

## 1. Timebase Normalization
As implemented in `rosbag_reader.py`:
*   **Per-Camera Zero**: We define $t=0$ for *each* camera as the timestamp of its first valid frame.
*   **Header Stamps**: We rely on `msg.header.stamp` (ROS time) when available, rather than bag receive time, to minimize jitter from USB transport.

## 2. Offset Estimation
To align streams $C_1, C_2, C_3$:
1.  We pick a **Master Camera** (usually the one with the most stable detections, e.g., $C_1$).
2.  We perform a coarse alignment using cross-correlation of the vertical displacement signal $y(t)$ (since gravity affects all cameras similarly).
3.  We find the time shift $\delta_i$ that maximizes correlation:
    $$ \tau_{opt} = \arg\max_\tau (y_{master} \star y_i)(\tau) $$

## 3. Resampling
Once offsets are known, we cannot simply array-slice because frame rates may fluctuate slightly.
*   **Master Grid**: We define a perfectly regular time grid at the target FPS (e.g., 60Hz: 0.000, 0.016, 0.033...).
*   **Interpolation**: We resample the pose trajectories of all cameras onto this master grid using **SLERP** (Spherical Linear Interpolation) for quaternions and linear interpolation for translation.
