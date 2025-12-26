# Data Schema

We adhere to a strict canonical schema for all CSV outputs to ensure compatibility between stages and legacy tools.

## Standard Columns

### Time & Identification
| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `t_sec` | float | s | Normalized time. Starts at 0.0 for the first frame. |
| `cam_id` | str | - | Camera identifier (e.g., `sony_cam1`). |
| `tag_id` | int | - | ID of the tracked AprilTag. |
| `frame_idx`| int | - | Original frame index from the camera stream. |

### 6DoF Pose (Camera Frame)
Pose of the Tag Frame expressed in Camera Frame.
| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `x_m` | float | m | Translation X. |
| `y_m` | float | m | Translation Y. |
| `z_m` | float | m | Translation Z (Depth). |
| `qx` | float | - | Quaternion X. |
| `qy` | float | - | Quaternion Y. |
| `qz` | float | - | Quaternion Z. |
| `qw` | float | - | Quaternion W (Scalar). |

### Quality Metrics (Paper 2)
| Column | Type | Description |
| :--- | :--- | :--- |
| `decision_margin` | float | Raw detection margin from AprilTag library. Higher is better. |
| `hamming` | int | Hamming distance error in tag decode. 0 is perfect. |
| `tag_area_px2` | float | Area of the tag in pixels. Used for tie-breaking. |
| `reproj_err_px` | float | Mean reprojection error of the 4 corners after `solvePnP`. |
| `detected` | int | 1 if object found, 0 if lost (blind frame). |
| `source` | str | Method used: `"tag"` (decode), `"klt"` (tracker), or `"none"`. |

## Legacy Compatibility
We maintain a legacy export function for compatibility with scripts from Paper 1 (IEEE Access).
*   **Renames**: `t_sec` -> `timestamp`, `x_m` -> `trans_x`, etc.
*   **Format**: Plain CSV without hierarchical headers.
