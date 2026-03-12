# PoseVision: Basketball Video Analytics

PoseVision is an end-to-end basketball video understanding pipeline that combines player detection, multi-object tracking, team classification, pose estimation, and structured JSON export.

This project was initially developed as an independent undergraduate research project and later expanded into a SURF research project.

## Project Overview

Modern sports analytics increasingly relies on automated video understanding. Basketball video analysis remains challenging because of:

- rapid player movement
- frequent occlusions
- multiple interacting objects
- complex tactical structures

PoseVision addresses these challenges by building a modular deep learning pipeline that extracts player trajectories, team labels, poses, and frame-level structured outputs from broadcast basketball footage.

## Current Pipeline

1. YOLO detects players and the ball.
2. DeepSORT keeps player and ball identities stable across frames.
3. A jersey-color clustering module assigns each player to one of two teams.
4. RTMPose predicts body keypoints for tracked players.
5. The system exports frame-by-frame results to JSON for later visualization and analysis.

## Architecture Diagram

![Pipeline](assets/pipeline.jpg)

## Demo Assets

### Player Detection and Tracking

![Tracking Demo](assets/tracking.gif)

### Pose Estimation

![Pose Demo](assets/pose.gif)

### 2D Tactical Projection

![Projection Demo](assets/projection.gif)

## Dataset

Dataset page:
- [PoseVision dataset](https://huggingface.co/datasets/YingboJiao22/PoseVision)

Current dataset summary:
- Resolution: 3840x2160
- Frame rate: 50 FPS
- Total frames: about 30,000
- Annotated frames: 2,000
- Annotation categories: players, referees, basketball, ball

## Experimental Results

Current reported results:

| Module | Metric | Result |
|------|------|------|
| Detection | mAP | 96.1% |
| Tracking | MOTA | 82.1% |
| Team classification | Accuracy | about 95% |
| Pose estimation | PCK | above 83% |

These numbers still need to be fully documented in the thesis with experiment settings and evaluation details.

## Repository Structure

```text
PoseVision/
├── assets/
├── classify/
├── configs/
├── court/
├── input/
├── output/
├── pose/
├── tracking/
├── utils/
├── weights/
├── yolo_detection/
├── main.py
├── requirements.txt
└── README.md
```

## Important Files

- `main.py`: main command-line entry for the full pipeline
- `classify/team_classifier.py`: reusable team classification module used by the pipeline
- `classify/classify.py`: legacy experimental script kept only for reference
- `input/README.md`: where to place your local test videos

## Environment

- Python 3.10 or newer is recommended.
- A CUDA-capable GPU is recommended for RTMPose inference.
- If you do not have a GPU, set `--device cpu` when running the pipeline.

## Installation

```bash
pip install -r requirements.txt
```

## Required Files

Before running the project, make sure these files exist:

- YOLO weights: `yolo_detection/best.pt`
- RTMPose config: `configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py`
- RTMPose checkpoint: `configs/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth`
- Input video: place one test video in `input/`, for example `input/sample.mp4`

## Quick Start

Run the full pipeline:

```bash
python main.py --video input/sample.mp4
```

If you want CPU inference:

```bash
python main.py --video input/sample.mp4 --device cpu
```

If you want a custom output location:

```bash
python main.py --video input/sample.mp4 --output-json output/analysis_results.json
```

## Outputs

After running the pipeline, the main outputs are written to `output/`:

- `analysis_results.json`: tracked players, tracked balls, and pose results
- `classified_results.json`: per-frame team assignments for player detections

## Visualization Scripts

Draw team-color bounding boxes:

```bash
python utils/vis_classify.py
```

Draw pose results on a black background:

```bash
python utils/vis_skeleton.py
```

## Notes for Graduation Project Use

The main pipeline has been cleaned up to use:

- relative paths instead of machine-specific absolute paths
- a command-line entry point
- a reusable `TeamClassifier` interface
- a consistent `classified_results.json` format for downstream visualization

Before final thesis submission, you should still add:

- dataset preparation details
- metric definitions and experiment settings
- ablation or comparison experiments
- failure case analysis
