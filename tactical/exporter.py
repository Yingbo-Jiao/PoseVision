from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TacticalDataExporter:
    def __init__(self, court, ball_interpolation_max_gap: int = 12, ball_smoothing_alpha: float = 0.35):
        self.court = court
        self.ball_interpolation_max_gap = ball_interpolation_max_gap
        self.ball_smoothing_alpha = ball_smoothing_alpha

    def build_tactical_data(self, analysis_results: dict[str, Any], fps: float | None = None) -> dict[str, Any]:
        frame_keys = sorted(analysis_results.keys(), key=lambda key: int(key.split("_")[1]))
        ball_candidates = [self._project_ball_candidates(analysis_results[key]) for key in frame_keys]
        primary_ball_track = self._select_primary_ball_track(ball_candidates)
        primary_ball_track = self._interpolate_ball_track(primary_ball_track)
        primary_ball_track = self._smooth_ball_track(primary_ball_track)

        data = {
            "metadata": {
                "court_width": self.court.calibration.court_width,
                "court_height": self.court.calibration.court_height,
                "court_mode": self.court.calibration.court_mode,
                "total_frames": len(frame_keys),
                "fps": fps,
                "ball_interpolation_max_gap": self.ball_interpolation_max_gap,
                "ball_smoothing_alpha": self.ball_smoothing_alpha,
            },
            "frames": {},
        }

        for index, frame_key in enumerate(frame_keys):
            frame_data = analysis_results[frame_key]
            team_buckets = {"0": [], "1": [], "unknown": []}
            projected_balls = []

            for player in frame_data.get("players", []):
                bbox = player.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                projected = self.court.project_point((x1 + x2) / 2.0, y2)
                if projected is None:
                    continue
                court_x, court_y = projected
                player_entry = {
                    "player_id": player.get("id", -1),
                    "team_id": int(player.get("team", -1)),
                    "court_x": court_x,
                    "court_y": court_y,
                    "original_x": float((x1 + x2) / 2.0),
                    "original_y": float(y2),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "pose": player.get("pose"),
                    "conf": player.get("conf"),
                }
                if player_entry["team_id"] == 0:
                    team_buckets["0"].append(player_entry)
                elif player_entry["team_id"] == 1:
                    team_buckets["1"].append(player_entry)
                else:
                    team_buckets["unknown"].append(player_entry)

            for ball in frame_data.get("balls", []):
                bbox = ball.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                projected = self.court.project_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                if projected is None:
                    continue
                court_x, court_y = projected
                projected_balls.append(
                    {
                        "ball_id": ball.get("id", -1),
                        "court_x": court_x,
                        "court_y": court_y,
                        "original_x": float((x1 + x2) / 2.0),
                        "original_y": float((y1 + y2) / 2.0),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

            data["frames"][frame_key] = {
                "frame_index": index + 1,
                "ball": primary_ball_track[index],
                "projected_balls": projected_balls,
                "teams": {
                    "0": self._build_team_payload(team_buckets["0"]),
                    "1": self._build_team_payload(team_buckets["1"]),
                    "unknown": self._build_team_payload(team_buckets["unknown"]),
                },
            }

        return data

    def export(self, analysis_results: dict[str, Any], output_path: str | Path, fps: float | None = None) -> Path:
        data = self.build_tactical_data(analysis_results, fps=fps)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[INFO] Saved tactical-ready data to {output_path}")
        return output_path

    def export_master_sequence(
        self,
        analysis_results: dict[str, Any],
        classified_frames: list[dict[str, Any]],
        output_path: str | Path,
        fps: float | None = None,
    ) -> Path:
        tactical_data = self.build_tactical_data(analysis_results, fps=fps)
        classified_lookup = {item["frame"]: item for item in classified_frames}

        master = {
            "metadata": {
                "video": {
                    "fps": fps,
                    "total_frames": len(analysis_results),
                },
                "court": {
                    "mode": self.court.calibration.court_mode,
                    "width": self.court.calibration.court_width,
                    "height": self.court.calibration.court_height,
                    "image_points": self.court.calibration.image_points,
                    "reference_points": self.court.calibration.reference_points,
                    "roi_points": self.court.calibration.roi_points,
                },
                "ball_processing": {
                    "interpolation_max_gap": self.ball_interpolation_max_gap,
                    "smoothing_alpha": self.ball_smoothing_alpha,
                },
            },
            "frames": {},
        }

        frame_keys = sorted(analysis_results.keys(), key=lambda key: int(key.split("_")[1]))
        for frame_key in frame_keys:
            frame_index = int(frame_key.split("_")[1])
            tactical_frame = tactical_data["frames"][frame_key]
            analysis_frame = analysis_results[frame_key]
            classified_frame = classified_lookup.get(frame_index, {"players": [], "original_player_dets": []})

            master["frames"][frame_key] = {
                "frame_index": frame_index,
                "analysis": {
                    "players": analysis_frame.get("players", []),
                    "balls": analysis_frame.get("balls", []),
                },
                "classification": {
                    "players": classified_frame.get("players", []),
                    "original_player_dets": classified_frame.get("original_player_dets", []),
                },
                "projection": {
                    "ball": tactical_frame["ball"],
                    "projected_balls": tactical_frame.get("projected_balls", []),
                    "teams": tactical_frame["teams"],
                },
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(master, indent=2), encoding="utf-8")
        print(f"[INFO] Saved master sequence data to {output_path}")
        return output_path

    def _project_ball_candidates(self, frame_data: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for ball in frame_data.get("balls", []):
            bbox = ball.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            projected = self.court.project_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if projected is None:
                continue
            court_x, court_y = projected
            candidates.append(
                {
                    "visible": True,
                    "source": "observed",
                    "ball_id": ball.get("id", -1),
                    "court_x": court_x,
                    "court_y": court_y,
                    "original_x": float((x1 + x2) / 2.0),
                    "original_y": float((y1 + y2) / 2.0),
                }
            )
        return candidates

    def _select_primary_ball_track(self, candidate_frames: list[list[dict[str, Any]]]) -> list[dict[str, Any] | None]:
        chosen: list[dict[str, Any] | None] = []
        previous = None
        for candidates in candidate_frames:
            if not candidates:
                chosen.append(None)
                continue
            if previous is None:
                selected = candidates[0]
            else:
                selected = min(
                    candidates,
                    key=lambda candidate: (candidate["court_x"] - previous["court_x"]) ** 2
                    + (candidate["court_y"] - previous["court_y"]) ** 2,
                )
            chosen.append(selected.copy())
            previous = selected
        return chosen

    def _interpolate_ball_track(self, track: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
        result = [item.copy() if item is not None else None for item in track]
        observed_indices = [index for index, item in enumerate(result) if item is not None]

        for left, right in zip(observed_indices, observed_indices[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > self.ball_interpolation_max_gap:
                continue
            start = result[left]
            end = result[right]
            assert start is not None and end is not None
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                result[left + offset] = {
                    "visible": True,
                    "source": "interpolated",
                    "ball_id": -1,
                    "court_x": float(start["court_x"] + ratio * (end["court_x"] - start["court_x"])),
                    "court_y": float(start["court_y"] + ratio * (end["court_y"] - start["court_y"])),
                    "original_x": float(start["original_x"] + ratio * (end["original_x"] - start["original_x"])),
                    "original_y": float(start["original_y"] + ratio * (end["original_y"] - start["original_y"])),
                }

        filled = []
        for item in result:
            if item is None:
                filled.append(
                    {
                        "visible": False,
                        "source": "missing",
                        "ball_id": -1,
                        "court_x": None,
                        "court_y": None,
                        "original_x": None,
                        "original_y": None,
                    }
                )
            else:
                filled.append(item)
        return filled

    def _smooth_ball_track(self, track: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previous = None
        smoothed = []
        for item in track:
            current = item.copy()
            if current["visible"] and current["court_x"] is not None and current["court_y"] is not None:
                if previous is None:
                    previous = current.copy()
                else:
                    alpha = self.ball_smoothing_alpha
                    current["court_x"] = float(alpha * current["court_x"] + (1 - alpha) * previous["court_x"])
                    current["court_y"] = float(alpha * current["court_y"] + (1 - alpha) * previous["court_y"])
                    if current["original_x"] is not None and previous["original_x"] is not None:
                        current["original_x"] = float(alpha * current["original_x"] + (1 - alpha) * previous["original_x"])
                        current["original_y"] = float(alpha * current["original_y"] + (1 - alpha) * previous["original_y"])
                    previous = current.copy()
            smoothed.append(current)
        return smoothed

    def _build_team_payload(self, players: list[dict[str, Any]]) -> dict[str, Any]:
        players = sorted(players, key=lambda item: (item["court_y"], item["court_x"]))
        points = np.asarray([[item["court_x"], item["court_y"]] for item in players], dtype=np.float32) if players else None
        centroid = None
        spread = None
        occupied_area = None
        if points is not None and len(points) > 0:
            centroid = {
                "court_x": float(points[:, 0].mean()),
                "court_y": float(points[:, 1].mean()),
            }
            spread = {
                "width": float(points[:, 0].max() - points[:, 0].min()) if len(points) > 0 else 0.0,
                "depth": float(points[:, 1].max() - points[:, 1].min()) if len(points) > 0 else 0.0,
            }
            if len(points) >= 3:
                hull = cv2.convexHull(points.reshape(-1, 1, 2))
                occupied_area = float(cv2.contourArea(hull))
            else:
                occupied_area = 0.0

        return {
            "count": len(players),
            "centroid": centroid,
            "spread": spread,
            "occupied_area": occupied_area,
            "players": players,
        }
