import cv2
import json
from tqdm import tqdm

def visualize_teams(video_path, classified_json_path, output_video_path):
    """
    在视频上绘制球队边界框：
    - team 0: 蓝色
    - team 1: 白色
    """

    # 读取分类结果
    with open(classified_json_path, "r") as f:
        classified_data = json.load(f)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 转换成 {frame_num: players_list} 方便查找
    frame_dict = {item["frame"]: item["players"] for item in classified_data}

    for frame_idx in tqdm(range(frame_count), desc="Drawing Teams"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx + 1  # JSON 的帧号是从 1 开始的
        players = frame_dict.get(frame_num, [])

        for player in players:
            x1, y1, x2, y2 = map(int, player["bbox"])
            team = player["team"]

            color = (255, 0, 0) if team == 0 else (255, 255, 255)  # 蓝 or 白
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"队伍边界框视频已保存到 {output_video_path}")


if __name__ == "__main__":
    video_path = r"C:\Users\Yingbo.Jiao\Desktop\sample1.mp4"
    classified_json_path = r"C:\Users\Yingbo.Jiao\Desktop\classified_results(1).json"
    output_video_path = r"C:\Users\Yingbo.Jiao\Desktop\PoseVision\output\team_visualization.mp4"

    visualize_teams(video_path, classified_json_path, output_video_path)
