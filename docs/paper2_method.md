# Paper 2: ProposedMethodology

This document details the algorithmic contributions for **Paper 2**, focusing on improving tracking robustness under challenging conditions (motion blur, occlusion) where standard AprilTag detectors fail.

## 1. Hybrid AprilTag + KLT Tracking

Standard AprilTag detectors (like `pupil_apriltags`) require sharp edges to decode the tag quad. Fast vibration often causes motion blur, leading to detection dropouts.

**Our Approach:**
1.  **Keyframe Detection**: When a tag is successfully detected, we cache its 4 corner locations in the image with sub-pixel precision.
2.  **KLT Fallback**: In subsequent frames where tag detection fails:
    *   Initialize Lucas-Kanade (LK) optical flow trackers on the last known corner positions.
    *   Track these corners forward in time.
    *   **Verification**: Check geometric consistency (area, aspect ratio) to reject tracking drift.
3.  **Pose Recovery**: Use `solvePnP` on the tracked corners to recover the 6DoF pose, even without a valid tag decode.

## 2. Motion Consistency Index (MCI)

To validate pose updates without ground truth, we introduce the **MCI**.

*   **Concept**: Comparing the velocity implied by the pose change $\Delta P$ against an independent visual measurement (dense optical flow).
*   **Metric**:
    $MCI(t) = | v_{pose}(t) - v_{flow}(t) |$
*   **Logic**:
    *   If $MCI$ is low, the pose update is consistent with the visual motion field $\rightarrow$ **Accept**.
    *   If $MCI$ is high, the pose update is likely an outlier (e.g., corner swap, tracking drift) $\rightarrow$ **Reject/Downweight**.

## 3. Confidence-Weighted Fusion

We fuse estimates from $N$ cameras:

$$ T_{fused}(t) = \frac{\sum w_i \cdot T_i(t)}{\sum w_i} $$

**Weights $w_i$** are derived from:
*   **Detection Margin**: Stronger signal = higher weight.
*   **Reprojection Error**: Better fit = higher weight.
*   **KLT Confidence**: Tracked frames have lower weight than fully decoded tags.
*   **MCI Score**: Inconsistent frames are heavily penalized.
