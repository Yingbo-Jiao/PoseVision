import cv2
import json
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "input" / "sample.mp4"
DEFAULT_CLASSIFIED_JSON = PROJECT_ROOT / "output" / "classified_results.json"
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "output" / "team_visualization.mp4"


def visualize_teams(video_path, classified_json_path, output_video_path):
    """
    Draw team-colored bounding boxes on a video.
    - team 0: blue
    - team 1: white
    """

    with open(classified_json_path, "r", encoding="utf-8") as f:
        classified_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_dict = {item["frame"]: item["players"] for item in classified_data}

    for frame_idx in tqdm(range(frame_count), desc="Drawing Teams"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx + 1
        players = frame_dict.get(frame_num, [])

        for player in players:
            x1, y1, x2, y2 = map(int, player["bbox"])
            team = player.get("team", -1)
            color = (255, 0, 0) if team == 0 else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Team visualization saved to {output_path}")


if __name__ == "__main__":
    visualize_teams(
        video_path=str(DEFAULT_VIDEO),
        classified_json_path=str(DEFAULT_CLASSIFIED_JSON),
        output_video_path=str(DEFAULT_OUTPUT_VIDEO),
    )
