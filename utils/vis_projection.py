from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TACTICAL_JSON = PROJECT_ROOT / 'output' / 'tactical_sequences.json'
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / 'output' / 'projection_visualization.mp4'

TEAM_COLORS = {
    '0': (66, 135, 245),
    '1': (245, 245, 245),
    'unknown': (120, 120, 120),
}
BALL_COLOR = (52, 235, 183)
COURT_COLOR = (186, 140, 90)
COURT_LINE_COLOR = (245, 245, 245)
BACKGROUND_COLOR = (34, 89, 34)


def draw_court(canvas: np.ndarray, court_width: int, court_height: int, mode: str) -> np.ndarray:
    canvas[:] = BACKGROUND_COLOR
    margin = 30
    cv2.rectangle(canvas, (margin, margin), (court_width - margin, court_height - margin), COURT_COLOR, -1)
    cv2.rectangle(canvas, (margin, margin), (court_width - margin, court_height - margin), COURT_LINE_COLOR, 3)

    center_x = court_width // 2
    center_y = court_height // 2

    if mode == 'full':
        cv2.line(canvas, (center_x, margin), (center_x, court_height - margin), COURT_LINE_COLOR, 3)
        cv2.circle(canvas, (center_x, center_y), 90, COURT_LINE_COLOR, 3)

    for side in ('left', 'right') if mode == 'full' else ('left',):
        if side == 'left':
            base_x = margin
            hoop_x = margin + 65
            lane_outer_x = margin + 190
            lane_inner_x = margin + 110
            arc_center = (hoop_x, center_y)
            three_p1 = (margin, center_y - 250)
            three_p2 = (margin, center_y + 250)
            start_angle, end_angle = -72, 72
        else:
            base_x = court_width - margin
            hoop_x = court_width - margin - 65
            lane_outer_x = court_width - margin - 190
            lane_inner_x = court_width - margin - 110
            arc_center = (hoop_x, center_y)
            three_p1 = (court_width - margin, center_y - 250)
            three_p2 = (court_width - margin, center_y + 250)
            start_angle, end_angle = 108, 252

        cv2.circle(canvas, (hoop_x, center_y), 8, COURT_LINE_COLOR, 2)
        cv2.rectangle(
            canvas,
            (min(base_x, lane_outer_x), center_y - 180),
            (max(base_x, lane_outer_x), center_y + 180),
            COURT_LINE_COLOR,
            3,
        )
        cv2.rectangle(
            canvas,
            (min(base_x, lane_inner_x), center_y - 90),
            (max(base_x, lane_inner_x), center_y + 90),
            COURT_LINE_COLOR,
            3,
        )
        cv2.ellipse(canvas, arc_center, (240, 240), 0, start_angle, end_angle, COURT_LINE_COLOR, 3)
        cv2.line(canvas, three_p1, (three_p1[0] + (0 if side == 'left' else 0), three_p1[1]), COURT_LINE_COLOR, 3)
        cv2.line(canvas, three_p2, (three_p2[0] + (0 if side == 'left' else 0), three_p2[1]), COURT_LINE_COLOR, 3)

    return canvas


def court_to_canvas(x: float, y: float, width: int, height: int, canvas_width: int, canvas_height: int) -> tuple[int, int]:
    cx = int(np.clip(round((x / max(width, 1)) * (canvas_width - 1)), 0, canvas_width - 1))
    cy = int(np.clip(round((y / max(height, 1)) * (canvas_height - 1)), 0, canvas_height - 1))
    return cx, cy


def draw_team_points(frame_canvas: np.ndarray, team_payload: dict, team_key: str, court_width: int, court_height: int):
    color = TEAM_COLORS[team_key]
    for player in team_payload.get('players', []):
        if player.get('court_x') is None or player.get('court_y') is None:
            continue
        px, py = court_to_canvas(player['court_x'], player['court_y'], court_width, court_height, frame_canvas.shape[1], frame_canvas.shape[0])
        cv2.circle(frame_canvas, (px, py), 11 if team_key != 'unknown' else 8, color, -1)
        cv2.circle(frame_canvas, (px, py), 11 if team_key != 'unknown' else 8, (30, 30, 30), 2)

    centroid = team_payload.get('centroid')
    if centroid and centroid.get('x') is not None and centroid.get('y') is not None:
        cx, cy = court_to_canvas(centroid['x'], centroid['y'], court_width, court_height, frame_canvas.shape[1], frame_canvas.shape[0])
        cv2.drawMarker(frame_canvas, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)


def draw_ball(frame_canvas: np.ndarray, ball_payload: dict, trail: deque, court_width: int, court_height: int):
    if not ball_payload.get('visible'):
        return
    if ball_payload.get('court_x') is None or ball_payload.get('court_y') is None:
        return

    bx, by = court_to_canvas(ball_payload['court_x'], ball_payload['court_y'], court_width, court_height, frame_canvas.shape[1], frame_canvas.shape[0])
    trail.append((bx, by))

    for idx in range(1, len(trail)):
        alpha = idx / max(len(trail), 1)
        color = tuple(int(channel * alpha) for channel in BALL_COLOR)
        cv2.line(frame_canvas, trail[idx - 1], trail[idx], color, 2)

    radius = 10 if ball_payload.get('source') == 'observed' else 8
    cv2.circle(frame_canvas, (bx, by), radius, BALL_COLOR, -1)
    cv2.circle(frame_canvas, (bx, by), radius, (30, 30, 30), 2)


def draw_overlay(frame_canvas: np.ndarray, frame_key: str, frame_payload: dict):
    teams = frame_payload.get('teams', {})
    ball = frame_payload.get('ball', {})
    lines = [
        f'{frame_key}',
        f"team0: {teams.get('0', {}).get('count', 0)}",
        f"team1: {teams.get('1', {}).get('count', 0)}",
        f"unknown: {teams.get('unknown', {}).get('count', 0)}",
        f"ball: {ball.get('source', 'missing')}",
    ]

    panel_w = 250
    cv2.rectangle(frame_canvas, (18, 18), (18 + panel_w, 18 + 28 * len(lines) + 16), (18, 18, 18), -1)
    cv2.rectangle(frame_canvas, (18, 18), (18 + panel_w, 18 + 28 * len(lines) + 16), (220, 220, 220), 2)

    for idx, text in enumerate(lines):
        cv2.putText(
            frame_canvas,
            text,
            (32, 50 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def visualize_projection(tactical_json_path: str | Path, output_video_path: str | Path, trail_length: int = 20, fps_override: float | None = None):
    tactical_json_path = Path(tactical_json_path)
    output_video_path = Path(output_video_path)

    data = json.loads(tactical_json_path.read_text(encoding='utf-8'))
    metadata = data.get('metadata', {})
    frames = data.get('frames', {})

    court_width = int(metadata.get('court_width', 1400))
    court_height = int(metadata.get('court_height', 800))
    court_mode = metadata.get('court_mode', 'full')
    fps = float(fps_override or metadata.get('fps') or 25.0)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (court_width, court_height))

    frame_keys = sorted(frames.keys(), key=lambda key: int(key.split('_')[1]))
    trail: deque[tuple[int, int]] = deque(maxlen=trail_length)

    for frame_key in tqdm(frame_keys, desc='Drawing Projection'):
        canvas = np.zeros((court_height, court_width, 3), dtype=np.uint8)
        draw_court(canvas, court_width, court_height, court_mode)

        frame_payload = frames[frame_key]
        teams = frame_payload.get('teams', {})
        draw_team_points(canvas, teams.get('0', {}), '0', court_width, court_height)
        draw_team_points(canvas, teams.get('1', {}), '1', court_width, court_height)
        draw_team_points(canvas, teams.get('unknown', {}), 'unknown', court_width, court_height)
        draw_ball(canvas, frame_payload.get('ball', {}), trail, court_width, court_height)
        draw_overlay(canvas, frame_key, frame_payload)

        writer.write(canvas)

    writer.release()
    print(f'Projection visualization saved to {output_video_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize 2D tactical projection data.')
    parser.add_argument('--tactical-json', default=str(DEFAULT_TACTICAL_JSON), help='Path to tactical_sequences.json')
    parser.add_argument('--output-video', default=str(DEFAULT_OUTPUT_VIDEO), help='Path to save the projection video')
    parser.add_argument('--trail-length', type=int, default=20, help='Ball trail length in frames')
    parser.add_argument('--fps', type=float, default=None, help='Override output FPS')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_projection(
        tactical_json_path=args.tactical_json,
        output_video_path=args.output_video,
        trail_length=args.trail_length,
        fps_override=args.fps,
    )
