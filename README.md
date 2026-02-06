# Indoor Visual SLAM Mapper
A step-by-step implementation of a monocular Visual SLAM system for indoor mapping.
## Project Overview
This project implements a complete visual perception pipeline for simultaneous localization and mapping (SLAM) using a single camera (laptop webcam or smartphone).
## Directory Structure
indoor_slam_mapper/
|--- src/
│   |-- phase1_camera.py          # Phase 1: Camera acquisition & calibration **in progress**
│   |-- phase2_features.py        # Phase 2: Feature detection & extraction **in progress**
│   |-- phase3_matching.py        # Phase 3: Feature matching between frames **in progress**
│   |-- phase4_pose.py            # Phase 4: Pose estimation & tracking **in progress**
│   |-- phase5_mapping.py         # Phase 5: 3D point cloud mapping **in progress**
│   |-- phase6_loop_closure.py    # Phase 6: Loop closure detection **in progress**
│   |-- slam_system.py            # Complete integrated system **in progress**
│   |-- utils.py                  # Shared utility functions **in progress**
|--- config/
│   |-- camera_params.yaml        # Camera calibration parameters
│   |-- slam_config.yaml          # SLAM system configuration
|--- data/
│   |-- calibration/              # Camera calibration images
│   |-- test_videos/              # Test sequences
|--- tests/
│   |-- test_phases.py            # Unit tests for each phase
├── outputs/
│   |-- trajectories/             # Saved camera trajectories
│   |-- maps/                     # 3D point cloud maps
│   |-- visualizations/           # Result visualizations
├── docs/
│   └── phase_guides/             # Detailed guide for each phase
|-- README.md                     # This file

## Development Phases
### Phase 1: Camera Feed Acquisition
Goal: Establish reliable video input and camera calibration
Components:

Camera interface (webcam/video file)
Camera calibration routine
Frame preprocessing
Real-time display

Deliverable: Working camera feed with calibrated intrinsics
Test: Display live camera feed with calibration overlay

### Phase 2: Feature Detection
Goal: Detect and track visual features in frames
Components:

ORB feature detector implementation
Feature visualization
Feature quality metrics
Multi-scale detection

Deliverable: Real-time feature detection display
Test: Detect 500+ features per frame with good distribution

### Phase 3: Feature Matching
Goal: Establish correspondences between consecutive frames
Components:

Brute-force matcher
RANSAC outlier rejection
Match visualization
Temporal consistency checking

Deliverable: Robust feature tracking across frames
Test: Maintain 200+ matches with <5% outliers

### Phase 4: Pose Estimation
Goal: Estimate camera motion from feature matches
Components:

Essential matrix computation
Pose recovery (R, t)
Scale estimation
Trajectory visualization

Deliverable: 2D trajectory plot of camera motion
Test: Smooth trajectory with correct motion direction

### Phase 5: 3D Point Cloud Mapping
Goal: Build sparse 3D map of environment
Components:

Point triangulation
Bundle adjustment
Map point management
3D visualization

Deliverable: Interactive 3D point cloud
Test: Recognizable room structure in point cloud

### Phase 6: Loop Closure Detection
Goal: Detect revisited locations and correct drift
Components:

Keyframe database
Place recognition (Bag of Words)
Loop verification
Pose graph optimization

Deliverable: Drift-corrected mapping
Test: Detect loop when returning to start position