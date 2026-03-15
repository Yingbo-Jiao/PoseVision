from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon


COURT_PRESETS = {
    "half": (1400, 1500),
    "full": (1400, 2800),
}


@dataclass
class CourtCalibration:
    roi_points: list[list[float]]
    image_points: list[list[float]]
    reference_points: list[list[float]]
    court_width: int
    court_height: int
    court_mode: str

    def to_dict(self) -> dict:
        return {
            "roi_points": self.roi_points,
            "image_points": self.image_points,
            "reference_points": self.reference_points,
            "court_width": self.court_width,
            "court_height": self.court_height,
            "court_mode": self.court_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CourtCalibration":
        return cls(
            roi_points=data["roi_points"],
            image_points=data["image_points"],
            reference_points=data["reference_points"],
            court_width=int(data["court_width"]),
            court_height=int(data["court_height"]),
            court_mode=data.get("court_mode", "half"),
        )


class Court:
    def __init__(self, calibration_path=None, polygon_path=None):
        self.calibration_path = Path(calibration_path) if calibration_path else Path("court_calibration.json")
        self.polygon_path = Path(polygon_path) if polygon_path else Path("court_polygon.json")
        self.calibration: CourtCalibration | None = None
        self.polygon: Polygon | None = None
        self.homography_matrix: np.ndarray | None = None

    def load_calibration(self) -> bool:
        if not self.calibration_path.exists():
            return False

        data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        self.calibration = CourtCalibration.from_dict(data)
        self.polygon = Polygon(self.calibration.roi_points)
        self.homography_matrix = cv2.getPerspectiveTransform(
            np.asarray(self.calibration.image_points, dtype=np.float32),
            np.asarray(self.calibration.reference_points, dtype=np.float32),
        )
        print(f"[INFO] Loaded court calibration from {self.calibration_path}")
        return True

    def choose_court_mode(self, default_mode: str = "half") -> str:
        window_name = "Court Mode Selection"
        canvas = np.full((220, 700, 3), (35, 35, 35), dtype=np.uint8)
        cv2.putText(canvas, "Select reference court mode", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Press H for half court", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, "Press F for full court", (40, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, f"Default: {default_mode}", (470, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, canvas)

        selected = None
        while selected is None:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("h"), ord("H")):
                selected = "half"
            elif key in (ord("f"), ord("F")):
                selected = "full"
            elif key == 27:
                cv2.destroyWindow(window_name)
                raise RuntimeError("Court mode selection cancelled by user.")
            elif key in (13, 10):
                selected = default_mode

        cv2.destroyWindow(window_name)
        return selected

    def calibrate(self, frame: np.ndarray, court_mode: str = "half") -> CourtCalibration:
        if court_mode not in COURT_PRESETS:
            raise ValueError(f"Unsupported court mode: {court_mode}")

        court_width, court_height = COURT_PRESETS[court_mode]
        print("[INFO] Court calibration started.")
        print("[INFO] Step 1: click 4 points on the video frame in this order: top-left, top-right, bottom-right, bottom-left.")
        print("[INFO] Step 2: click the matching 4 points on the reference court canvas in the same order.")
        print("[INFO] Press R to reset points, Enter to confirm after 4 clicks, Esc to cancel.")

        image_points = self._collect_points(
            frame.copy(),
            window_name="Video Calibration",
            instruction="Video: click TL -> TR -> BR -> BL | R reset | Enter confirm",
        )

        reference_canvas = self._draw_reference_court(court_mode, court_width, court_height)
        reference_points = self._collect_points(
            reference_canvas,
            window_name="Court Reference Calibration",
            instruction="Reference: click matching TL -> TR -> BR -> BL | R reset | Enter confirm",
        )

        self.calibration = CourtCalibration(
            roi_points=[[float(x), float(y)] for x, y in image_points],
            image_points=[[float(x), float(y)] for x, y in image_points],
            reference_points=[[float(x), float(y)] for x, y in reference_points],
            court_width=court_width,
            court_height=court_height,
            court_mode=court_mode,
        )
        self.polygon = Polygon(self.calibration.roi_points)
        self.homography_matrix = cv2.getPerspectiveTransform(
            np.asarray(self.calibration.image_points, dtype=np.float32),
            np.asarray(self.calibration.reference_points, dtype=np.float32),
        )
        self.save_calibration()
        return self.calibration

    def save_calibration(self) -> None:
        if self.calibration is None:
            raise ValueError("No court calibration available to save.")

        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(
            json.dumps(self.calibration.to_dict(), indent=2),
            encoding="utf-8",
        )
        self.polygon_path.parent.mkdir(parents=True, exist_ok=True)
        self.polygon_path.write_text(
            json.dumps(self.calibration.roi_points, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Saved court calibration to {self.calibration_path}")
        print(f"[INFO] Saved ROI polygon to {self.polygon_path}")

    def filter_detections_by_polygon(self, detections, image_shape):
        if self.polygon is None:
            return detections

        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            if int(class_id) == 1:
                filtered_detections.append(det)
                continue

            point = Point((x1 + x2) / 2.0, y2)
            if self.polygon.contains(point):
                filtered_detections.append(det)
        return filtered_detections

    def project_point(self, x: float, y: float) -> tuple[float, float] | None:
        if self.homography_matrix is None:
            return None

        points = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(points, self.homography_matrix)
        court_x, court_y = projected[0, 0].tolist()
        return float(court_x), float(court_y)

    def export_projected_results(self, analysis_results: dict, output_path: str | Path) -> Path:
        if self.calibration is None:
            raise ValueError("Court calibration is required before exporting projected results.")

        projected = {
            "court_dimensions": {
                "width": self.calibration.court_width,
                "height": self.calibration.court_height,
                "mode": self.calibration.court_mode,
            },
            "frames": {},
        }

        for frame_key, frame_data in analysis_results.items():
            projected_players = []
            projected_balls = []

            for player in frame_data.get("players", []):
                bbox = player.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                projected_point = self.project_point((x1 + x2) / 2.0, y2)
                if projected_point is None:
                    continue
                court_x, court_y = projected_point
                projected_players.append(
                    {
                        "player_id": player.get("id", -1),
                        "team_id": player.get("team", -1),
                        "court_x": court_x,
                        "court_y": court_y,
                        "original_x": float((x1 + x2) / 2.0),
                        "original_y": float(y2),
                    }
                )

            for ball in frame_data.get("balls", []):
                bbox = ball.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                projected_point = self.project_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                if projected_point is None:
                    continue
                court_x, court_y = projected_point
                projected_balls.append(
                    {
                        "ball_id": ball.get("id", -1),
                        "court_x": court_x,
                        "court_y": court_y,
                        "original_x": float((x1 + x2) / 2.0),
                        "original_y": float((y1 + y2) / 2.0),
                    }
                )

            projected["frames"][frame_key] = {
                "positions": projected_players,
                "balls": projected_balls,
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(projected, indent=2), encoding="utf-8")
        print(f"[INFO] Saved projected court positions to {output_path}")
        return output_path

    def _collect_points(self, frame: np.ndarray, window_name: str, instruction: str) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        labels = ["TL", "TR", "BR", "BL"]

        def redraw() -> None:
            canvas = frame.copy()
            cv2.putText(canvas, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 6, (0, 255, 0), -1)
                cv2.putText(canvas, labels[index], (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(points) == 4:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], True, (255, 255, 0), 2)
            cv2.imshow(window_name, canvas)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                redraw()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, click_event)
        redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if len(points) == 4:
                    break
            elif key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            elif key == 27:
                cv2.destroyWindow(window_name)
                raise RuntimeError("Court calibration cancelled by user.")

        cv2.destroyWindow(window_name)
        return points

    def _draw_reference_court(self, court_mode: str, width: int, height: int) -> np.ndarray:
        canvas = np.full((height, width, 3), (33, 105, 60), dtype=np.uint8)
        line_color = (240, 240, 240)
        line_thickness = 6
        margin = 40

        cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), line_color, line_thickness)

        if court_mode == "full":
            mid_y = height // 2
            cv2.line(canvas, (margin, mid_y), (width - margin, mid_y), line_color, 4)
            cv2.circle(canvas, (width // 2, mid_y), 90, line_color, 4)
            self._draw_half_court_markings(canvas, top=True, margin=margin, line_color=line_color)
            self._draw_half_court_markings(canvas, top=False, margin=margin, line_color=line_color)
        else:
            self._draw_half_court_markings(canvas, top=False, margin=margin, line_color=line_color)

        cv2.putText(canvas, f"Reference court: {court_mode}", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

    def _draw_half_court_markings(self, canvas: np.ndarray, top: bool, margin: int, line_color: tuple[int, int, int]) -> None:
        height, width = canvas.shape[:2]
        basket_x = width // 2
        if top:
            base_y = margin
            key_top = margin
            key_bottom = margin + 240
            hoop_y = margin + 70
            free_throw_y = margin + 190
            arc_center = (basket_x, hoop_y)
            arc_start, arc_end = 20, 160
        else:
            base_y = height - margin
            key_top = height - margin - 240
            key_bottom = height - margin
            hoop_y = height - margin - 70
            free_throw_y = height - margin - 190
            arc_center = (basket_x, hoop_y)
            arc_start, arc_end = 200, 340

        key_width = 320
        cv2.rectangle(canvas, (basket_x - key_width // 2, key_top), (basket_x + key_width // 2, key_bottom), line_color, 4)
        cv2.circle(canvas, (basket_x, free_throw_y), 90, line_color, 4)
        cv2.circle(canvas, (basket_x, hoop_y), 12, line_color, -1)
        cv2.ellipse(canvas, arc_center, (220, 220), 0, arc_start, arc_end, line_color, 4)
