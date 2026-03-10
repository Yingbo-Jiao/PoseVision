#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
from tqdm import tqdm

def visualize_pose_with_court(video_path, 
                              main_json_path,
                              output_video_path):
    """
    可视化处理流程：
    1. 黑色背景
    2. 绘制篮球场边界
    3. 绘制球员骨架（COCO 17点，pose字段）
    """

    # -------------------
    # 读取视频参数
    # -------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频: " + video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # -------------------
    # 读取 JSON
    # -------------------
    with open(main_json_path, 'r') as f:
        main_results = json.load(f)

    # -------------------
    # 视频写出器
    # -------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # -------------------
    # 篮球场边界四点
    # -------------------
    court_points = np.array([
        [956, 1125],
        [771, 1987],
        [3839, 1820],
        [3836, 1045]
    ], np.int32).reshape((-1, 1, 2))

    # -------------------
    # COCO17骨架连接规则
    # -------------------
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    # -------------------
    # 逐帧处理
    # -------------------
    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        # 黑色背景
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # 绘制场地边界
        cv2.polylines(frame, [court_points], isClosed=True, color=(0, 255, 255), thickness=4)

        # 获取JSON对应帧
        frame_key = f"frame_{frame_idx+1}"
        if frame_key in main_results:
            frame_data = main_results[frame_key]
            players = frame_data.get("players", [])

            for player in players:
                if "pose" in player:
                    kps = player["pose"]

                    # 先画点
                    for kp in kps:
                        if len(kp) == 2:   # [x,y]
                            x, y = kp
                            conf = 1.0
                        else:              # [x,y,conf]
                            x, y, conf = kp
                        if conf > 0.3:
                            cv2.circle(frame, (int(x), int(y)), radius=6, color=(0, 0, 255), thickness=-1)

                    # 再画骨架线
                    for (i, j) in skeleton:
                        if i < len(kps) and j < len(kps):
                            if len(kps[i]) == 2:
                                x1, y1 = kps[i]; c1 = 1.0
                            else:
                                x1, y1, c1 = kps[i]

                            if len(kps[j]) == 2:
                                x2, y2 = kps[j]; c2 = 1.0
                            else:
                                x2, y2, c2 = kps[j]

                            if c1 > 0.3 and c2 > 0.3:
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        out.write(frame)

    out.release()
    print("✅ 可视化完成，输出视频已保存到:", output_video_path)


if __name__ == "__main__":
    # 修改这里的路径即可运行
    video_path = r"C:\Users\Yingbo.Jiao\Desktop\sample1.mp4"  # 只用来取分辨率 & 帧数
    main_json_path = r'C:\Users\Yingbo.Jiao\Desktop\PoseVision\outputanalysis_results.json'
    output_video_path = r'C:\Users\Yingbo.Jiao\Desktop\PoseVision\output\pose_visualization_black.mp4'

    visualize_pose_with_court(video_path, main_json_path, output_video_path)
