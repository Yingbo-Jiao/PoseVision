from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from yolo_detection.detector import YOLODetector


@dataclass
class TeamClusterStats:
    saturation: float
    value: float


class TeamClassifier:
    """Cluster jersey appearance into two team labels."""

    def __init__(
        self,
        sample_video_path: str | None = None,
        detector: YOLODetector | None = None,
        random_state: int = 0,
        sample_frames: int = 48,
        min_detection_conf: float = 0.55,
    ) -> None:
        self.sample_video_path = sample_video_path
        self.detector = detector or YOLODetector()
        self.random_state = random_state
        self.sample_frames = sample_frames
        self.min_detection_conf = min_detection_conf
        self.kmeans = MiniBatchKMeans(n_clusters=2, random_state=random_state)
        self.is_fitted = False
        self.cluster_to_team: dict[int, int] = {0: 0, 1: 1}

    def classify_frame(self, frame: np.ndarray, player_detections: list[dict]) -> list[dict]:
        if not player_detections:
            return []

        if not self.is_fitted:
            self._bootstrap(frame, player_detections)

        features, valid_indices = self._extract_detection_features(frame, player_detections)
        if len(valid_indices) == 0:
            return [{**det, "team": -1} for det in player_detections]

        cluster_labels = self.kmeans.predict(features)
        classified = [{**det, "team": -1} for det in player_detections]
        for feature, det_index, cluster_label in zip(features, valid_indices, cluster_labels):
            team_id = self.cluster_to_team.get(int(cluster_label), int(cluster_label))
            classified[det_index]["team"] = int(team_id)
            classified[det_index]["appearance_feature"] = feature.tolist()

        return classified

    def _bootstrap(self, frame: np.ndarray, player_detections: list[dict]) -> None:
        sampled_features = []

        if self.sample_video_path:
            sampled_features.extend(self._collect_video_features(self.sample_video_path))

        if len(sampled_features) < 8:
            current_features, _ = self._extract_detection_features(frame, player_detections)
            sampled_features.extend(current_features.tolist())

        if len(sampled_features) < 2:
            raise ValueError("Unable to bootstrap team classifier from the provided video or frame.")

        feature_array = np.asarray(sampled_features, dtype=np.float32)
        self.kmeans.fit(feature_array)
        self.cluster_to_team = self._infer_team_mapping(feature_array)
        self.is_fitted = True

    def _collect_video_features(self, video_path: str) -> list[list[float]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        rng = random.Random(self.random_state)
        sample_count = min(self.sample_frames, total_frames)
        frame_indices = sorted(rng.sample(range(total_frames), sample_count))

        features: list[list[float]] = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue

            detections = self.detector.detect_image(frame)
            player_detections = [
                {
                    "bbox": [det[0], det[1], det[2], det[3]],
                    "conf": det[4],
                    "class_id": int(det[5]),
                }
                for det in detections
                if int(det[5]) == 0 and float(det[4]) >= self.min_detection_conf
            ]

            frame_features, _ = self._extract_detection_features(frame, player_detections)
            if len(frame_features) > 0:
                features.extend(frame_features.tolist())

        cap.release()
        return features

    def _extract_detection_features(
        self, frame: np.ndarray, player_detections: Iterable[dict]
    ) -> tuple[np.ndarray, list[int]]:
        features: list[np.ndarray] = []
        valid_indices: list[int] = []

        for index, det in enumerate(player_detections):
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            feature = self._extract_crop_feature(frame, bbox)
            if feature is None:
                continue
            features.append(feature)
            valid_indices.append(index)

        if not features:
            return np.empty((0, 20), dtype=np.float32), []

        return np.asarray(features, dtype=np.float32), valid_indices

    def _extract_crop_feature(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        upper_body = crop[: max(1, crop.shape[0] // 2), :]
        resized = cv2.resize(upper_body, (48, 96))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = value > 35
        if np.count_nonzero(mask) < 40:
            return None

        hue_hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [16], [0, 180]).flatten()
        hue_hist = hue_hist / (np.sum(hue_hist) + 1e-6)

        masked_sat = saturation[mask]
        masked_val = value[mask]
        stats = np.array(
            [
                float(masked_sat.mean()),
                float(masked_sat.std()),
                float(masked_val.mean()),
                float(masked_val.std()),
            ],
            dtype=np.float32,
        )

        return np.concatenate([hue_hist.astype(np.float32), stats], axis=0)

    def _infer_team_mapping(self, feature_array: np.ndarray) -> dict[int, int]:
        labels = self.kmeans.predict(feature_array)
        cluster_stats: dict[int, TeamClusterStats] = {}

        for cluster_id in range(2):
            cluster_features = feature_array[labels == cluster_id]
            if len(cluster_features) == 0:
                cluster_stats[cluster_id] = TeamClusterStats(saturation=0.0, value=0.0)
                continue
            cluster_stats[cluster_id] = TeamClusterStats(
                saturation=float(cluster_features[:, 16].mean()),
                value=float(cluster_features[:, 18].mean()),
            )

        white_cluster = min(
            cluster_stats,
            key=lambda cluster_id: (
                cluster_stats[cluster_id].saturation,
                -cluster_stats[cluster_id].value,
            ),
        )
        dark_cluster = 1 - white_cluster
        return {dark_cluster: 0, white_cluster: 1}
