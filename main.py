import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from classify.team_classifier import TeamClassifier
from pose.pose_estimator import PoseEstimator
from tracking.track import DeepSortTracker
from yolo_detection.detector import YOLODetector


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "yolo_detection" / "best.pt"
DEFAULT_POSE_CONFIG = (
    PROJECT_ROOT / "configs" / "body_2d_keypoint" / "rtmpose" / "body8" / "rtmpose-s_8xb256-420e_body8-256x192.py"
)
DEFAULT_POSE_CHECKPOINT = (
    PROJECT_ROOT / "configs" / "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
)


def analyze_video(
    video_path: str,
    output_json_path: str,
    yolo_weights: str,
    pose_config: str,
    pose_checkpoint: str,
    device: str = "cuda:0",
) -> tuple[Path, Path]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Unable to read first frame from: {video_path}")

    detector = YOLODetector(model_path=yolo_weights)
    team_classifier = TeamClassifier(sample_video_path=video_path, detector=detector)
    player_tracker = DeepSortTracker(max_age=30, n_init=3)
    ball_tracker = DeepSortTracker(max_age=5, n_init=2)
    pose_estimator = PoseEstimator(
        config_path=pose_config,
        checkpoint_path=pose_checkpoint,
        device=device,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = {}
    classified_frames = []

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect_image(frame)
        player_dets = []
        ball_dets = []
        for det in detections:
            detection = {
                "bbox": [det[0], det[1], det[2], det[3]],
                "conf": float(det[4]),
                "class_id": int(det[5]),
            }
            if detection["class_id"] == 0:
                player_dets.append(detection)
            elif detection["class_id"] == 1:
                ball_dets.append(detection)

        classified_players = team_classifier.classify_frame(frame, player_dets)
        tracked_players = player_tracker.update(frame, classified_players)
        tracked_balls = ball_tracker.update(frame, ball_dets)
        players_with_pose = pose_estimator.estimate(frame, tracked_players)

        frame_key = f"frame_{frame_idx + 1}"
        results[frame_key] = {
            "players": players_with_pose,
            "balls": tracked_balls,
        }
        classified_frames.append(
            {
                "frame": frame_idx + 1,
                "players": classified_players,
                "original_player_dets": player_dets,
            }
        )

    cap.release()

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path = output_path.parent / "classified_results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with classified_path.open("w", encoding="utf-8") as f:
        json.dump(classified_frames, f, indent=2)

    print(f"[INFO] Main analysis results saved to: {output_path}")
    print(f"[INFO] Team classification results saved to: {classified_path}")
    return output_path, classified_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PoseVision basketball video analysis pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_DIR / "analysis_results.json"),
        help="Path to the main analysis JSON output",
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(DEFAULT_YOLO_WEIGHTS),
        help="Path to the YOLO weights file",
    )
    parser.add_argument(
        "--pose-config",
        default=str(DEFAULT_POSE_CONFIG),
        help="Path to the RTMPose config file",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default=str(DEFAULT_POSE_CHECKPOINT),
        help="Path to the RTMPose checkpoint file",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    analyze_video(
        video_path=args.video,
        output_json_path=args.output_json,
        yolo_weights=args.yolo_weights,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
    )
