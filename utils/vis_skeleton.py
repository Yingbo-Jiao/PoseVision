#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "input" / "sample.mp4"
DEFAULT_MAIN_JSON = PROJECT_ROOT / "output" / "analysis_results.json"
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "output" / "pose_visualization_black.mp4"


def visualize_pose_with_court(video_path, main_json_path, output_video_path):
    """
    Visualize pose results on a black background with a court outline.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video: " + str(video_path))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    with open(main_json_path, 'r', encoding='utf-8') as f:
        main_results = json.load(f)

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    court_points = np.array([
        [956, 1125],
        [771, 1987],
        [3839, 1820],
        [3836, 1045]
    ], np.int32).reshape((-1, 1, 2))

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.polylines(frame, [court_points], isClosed=True, color=(0, 255, 255), thickness=4)

        frame_key = f"frame_{frame_idx + 1}"
        if frame_key in main_results:
            frame_data = main_results[frame_key]
            players = frame_data.get("players", [])

            for player in players:
                if "pose" not in player:
                    continue
                kps = player["pose"]

                for kp in kps:
                    if len(kp) == 2:
                        x, y = kp
                        conf = 1.0
                    else:
                        x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), radius=6, color=(0, 0, 255), thickness=-1)

                for i, j in skeleton:
                    if i >= len(kps) or j >= len(kps):
                        continue

                    if len(kps[i]) == 2:
                        x1, y1 = kps[i]
                        c1 = 1.0
                    else:
                        x1, y1, c1 = kps[i]

                    if len(kps[j]) == 2:
                        x2, y2 = kps[j]
                        c2 = 1.0
                    else:
                        x2, y2, c2 = kps[j]

                    if c1 > 0.3 and c2 > 0.3:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        out.write(frame)

    out.release()
    print("Pose visualization saved to:", output_path)


if __name__ == "__main__":
    visualize_pose_with_court(
        video_path=str(DEFAULT_VIDEO),
        main_json_path=str(DEFAULT_MAIN_JSON),
        output_video_path=str(DEFAULT_OUTPUT_VIDEO),
    )
