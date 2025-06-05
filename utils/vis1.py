import cv2
import json
import numpy as np
from tqdm import tqdm
import os

# 配置参数
JSON_PATH = r'C:\Users\Yingbo.Jiao22\Desktop\surf\output\players.json'  
VIDEO_PATH = r"C:\Users\Yingbo.Jiao22\Desktop\surf\input\sample1.mp4"  
OUTPUT_PATH = 'output/vis1_video.mp4'
CONF_THRESHOLD = 0.4  # 关键点置信度阈值

# COCO关键点连接关系
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (5, 11), 
    (6, 8), (6, 12), (7, 9), (8, 10), (11, 13), 
    (12, 14), (13, 15), (14, 16), (5, 6), (11, 12)
]

# 颜色配置
COLOR_MAP = {
    'A': (0, 255, 0),     # 绿色 - A队
    'B': (0, 0, 255),     # 红色 - B队
    'coach': (255, 0, 255) # 紫色 - 教练
}

# 可视化参数配置
VISUAL_CONFIG = {
    'keypoint_radius': 6,
    'skeleton_thickness': 3,
    'bbox_thickness': 2,
    'id_font_scale': 1.2,
    'info_font_scale': 1.5
}

# 加载JSON数据
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# 视频处理初始化
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 处理每帧数据
for frame_index in tqdm(range(total_frames), desc="Processing Video"):
    ret, frame = cap.read()
    if not ret:
        break

    # 获取当前帧的JSON数据
    frame_data = next((item for item in data if item["frame_id"] == frame_index + 1), None)
    if not frame_data:
        out.write(frame)
        continue

    # 绘制所有检测对象
    for team in ['A_team', 'B_team', 'coach']:
        for player in frame_data.get(team, []):
            # 解析数据
            track_id = player["track_id"]
            category = player["category"]
            keypoints = np.array(player["keypoints"][0])  # 取第一个关键点集合
            bbox = player["bbox"]

            # 获取颜色
            color = COLOR_MAP.get(category, (255, 255, 255))

            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, VISUAL_CONFIG['bbox_thickness'])

            # 绘制关键点
            for kp in keypoints:
                x, y = map(int, kp)
                cv2.circle(frame, (x, y), 
                          VISUAL_CONFIG['keypoint_radius'], 
                          color, -1)

            # 绘制骨架连接
            for (start, end) in skeleton:
                if start < len(keypoints) and end < len(keypoints):
                    x1, y1 = map(int, keypoints[start])
                    x2, y2 = map(int, keypoints[end])
                    cv2.line(frame, (x1, y1), (x2, y2),
                             color, VISUAL_CONFIG['skeleton_thickness'])

            # 显示追踪ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       VISUAL_CONFIG['id_font_scale'], 
                       color, 2)

    # 添加信息面板
    info_text = f"Frame: {frame_index + 1} | A: {len(frame_data['A_team'])} | B: {len(frame_data['B_team'])}"
    cv2.putText(frame, info_text, (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 
               VISUAL_CONFIG['info_font_scale'],
               (255, 255, 255), 3)

    # 写入视频帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
print(f"处理完成！输出视频已保存至：{OUTPUT_PATH}")