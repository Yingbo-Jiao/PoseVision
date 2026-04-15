from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / 'input' / 'sample2.mp4'
DEFAULT_MAIN_JSON = PROJECT_ROOT / 'output' / 'analysis_results.json'
DEFAULT_COURT_JSON = PROJECT_ROOT / 'output' / 'court_calibration.json'
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / 'output' / 'pose_visualization_black.mp4'

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_court_polygon(court_json_path: str | Path):
    court_path = Path(court_json_path)
    if not court_path.exists():
        return None
    data = load_json(court_path)
    points = data.get('roi_points') or data.get('image_points')
    if not points:
        return None
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def point_inside_expanded_box(point_x: float, point_y: float, bbox: list[float], padding_ratio: float) -> bool:
    x1, y1, x2, y2 = bbox
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return (x1 - pad_x) <= point_x <= (x2 + pad_x) and (y1 - pad_y) <= point_y <= (y2 + pad_y)


def normalize_keypoint(kp):
    if len(kp) >= 3:
        return float(kp[0]), float(kp[1]), float(kp[2])
    return float(kp[0]), float(kp[1]), 1.0


def build_valid_keypoints(player: dict, conf_threshold: float, bbox_padding_ratio: float):
    pose = player.get('pose')
    bbox = player.get('bbox')
    if not pose or not bbox or len(bbox) != 4:
        return None, None

    bbox = [float(v) for v in bbox]
    valid = []
    visible_count = 0
    for kp in pose:
        x, y, conf = normalize_keypoint(kp)
        is_valid = conf >= conf_threshold and point_inside_expanded_box(x, y, bbox, bbox_padding_ratio)
        valid.append((x, y, conf, is_valid))
        if is_valid:
            visible_count += 1

    return bbox, valid if visible_count >= 4 else None


def draw_player_pose(frame: np.ndarray, player: dict, conf_threshold: float, bbox_padding_ratio: float, max_limb_ratio: float, draw_bbox: bool):
    bbox, keypoints = build_valid_keypoints(player, conf_threshold, bbox_padding_ratio)
    if bbox is None or keypoints is None:
        return False

    x1, y1, x2, y2 = bbox
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)
    bbox_diag = float(np.hypot(bbox_w, bbox_h))
    max_limb_length = bbox_diag * max_limb_ratio

    if draw_bbox:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (90, 90, 90), 1)

    for i, j in SKELETON:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1_kp, y1_kp, c1, valid1 = keypoints[i]
        x2_kp, y2_kp, c2, valid2 = keypoints[j]
        if not (valid1 and valid2):
            continue
        limb_length = float(np.hypot(x2_kp - x1_kp, y2_kp - y1_kp))
        if limb_length > max_limb_length:
            continue
        cv2.line(frame, (int(x1_kp), int(y1_kp)), (int(x2_kp), int(y2_kp)), (0, 220, 0), 3)

    for x, y, conf, is_valid in keypoints:
        if not is_valid:
            continue
        radius = 5 if conf < 0.6 else 6
        cv2.circle(frame, (int(x), int(y)), radius=radius, color=(0, 80, 255), thickness=-1)

    return True


def visualize_pose_with_court(video_path, main_json_path, output_video_path, court_json_path=None, conf_threshold=0.35, bbox_padding_ratio=0.18, max_limb_ratio=0.65, draw_bbox=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Unable to open video: ' + str(video_path))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    main_results = load_json(main_json_path)
    court_polygon = load_court_polygon(court_json_path) if court_json_path else None

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    for frame_idx in tqdm(range(total_frames), desc='Processing Pose Video'):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame_key = f'frame_{frame_idx + 1}'
        frame_data = main_results.get(frame_key, {})
        players = frame_data.get('players', [])

        if court_polygon is not None:
            cv2.polylines(frame, [court_polygon], isClosed=True, color=(0, 255, 255), thickness=3)

        drawn_players = 0
        for player in players:
            if draw_player_pose(frame, player, conf_threshold, bbox_padding_ratio, max_limb_ratio, draw_bbox):
                drawn_players += 1

        cv2.putText(frame, frame_key, (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f'poses: {drawn_players}', (32, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'conf>{conf_threshold:.2f}  limb<{max_limb_ratio:.2f}diag', (32, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print('Pose visualization saved to:', output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize pose results on a black background.')
    parser.add_argument('--video', default=str(DEFAULT_VIDEO), help='Input video path')
    parser.add_argument('--main-json', default=str(DEFAULT_MAIN_JSON), help='Path to analysis_results.json')
    parser.add_argument('--court-json', default=str(DEFAULT_COURT_JSON), help='Path to court_calibration.json')
    parser.add_argument('--output-video', default=str(DEFAULT_OUTPUT_VIDEO), help='Output video path')
    parser.add_argument('--conf-threshold', type=float, default=0.35, help='Minimum keypoint confidence')
    parser.add_argument('--bbox-padding-ratio', type=float, default=0.18, help='Allowed keypoint padding outside bbox')
    parser.add_argument('--max-limb-ratio', type=float, default=0.65, help='Maximum limb length relative to bbox diagonal')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw player bounding boxes for debugging')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_pose_with_court(
        video_path=args.video,
        main_json_path=args.main_json,
        output_video_path=args.output_video,
        court_json_path=args.court_json,
        conf_threshold=args.conf_threshold,
        bbox_padding_ratio=args.bbox_padding_ratio,
        max_limb_ratio=args.max_limb_ratio,
        draw_bbox=args.draw_bbox,
    )
