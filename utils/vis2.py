import cv2
import json
import os
from collections import defaultdict
from tqdm import tqdm

def visualize(video_path, players_json, balls_json, output_path):
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化视频写入器
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        print(f"无法创建输出视频文件: {output_path}")
        cap.release()
        return

    # 加载JSON数据
    print("正在加载球员数据...")
    try:
        with open(players_json, 'r') as f:
            players = json.load(f)
        with open(balls_json, 'r') as f:
            balls = json.load(f)
    except Exception as e:
        print(f"加载JSON数据失败: {e}")
        cap.release()
        if 'writer' in locals():
            writer.release()
        return

    # 组织数据为按帧索引的字典
    print("正在组织数据...")
    players_by_frame = defaultdict(list)
    for p in tqdm(players, desc="处理球员数据"):
        try:
            players_by_frame[p['frame_id']].append(p)
        except KeyError:
            continue
    
    balls_by_frame = defaultdict(list)
    for b in tqdm(balls, desc="处理篮球数据"):
        try:
            balls_by_frame[b['frame_id']].append(b)
        except KeyError:
            continue

    # 颜色映射
    color_map = {
        'A': (39, 62, 109),        # 蓝色 - A队
        'B': (191, 220, 233),      # 浅蓝/白色 - B队
        'referee': (28, 35, 34),   # 黑色 - 裁判
        'sports ball': (0, 165, 255)  # 橙色 - 篮球
    }

    # 处理每一帧
    print("开始处理视频帧...")
    progress_bar = tqdm(total=total_frames, desc="处理视频进度", unit="帧")
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 绘制篮球
        for ball in balls_by_frame.get(frame_id, []):
            try:
                x1, y1, x2, y2 = map(int, ball['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_map['sports ball'], 2)
                cv2.putText(frame, f"Ball", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           color_map['sports ball'], 2)
            except (KeyError, ValueError) as e:
                continue

        # 绘制球员/裁判
        for player_data in players_by_frame.get(frame_id, []):
            for team in ['A_team', 'B_team', 'referee']:
                for person in player_data.get(team, []):
                    try:
                        # 绘制边界框
                        x1, y1, x2, y2 = map(int, person['bbox'])
                        color = color_map[person['category']]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 显示ID和类别
                        cv2.putText(frame, 
                                   f"ID:{person['track_id']}", 
                                   (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, color, 2)
                        
                        # 绘制关键点（如果存在）
                        if person.get('keypoints'):
                            for kp in person['keypoints']:
                                if isinstance(kp, list) and len(kp) >= 2:
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(frame, (x, y), 3, color, -1)
                    except (KeyError, ValueError, TypeError) as e:
                        continue

        writer.write(frame)
        progress_bar.update(1)

    # 释放资源
    progress_bar.close()
    cap.release()
    writer.release()
    print(f"\n可视化完成，结果已保存至 {output_path}")

# 使用示例
if __name__ == "__main__":
    video_path = r'C:\Users\Yingbo.Jiao22\Desktop\surf\input\sample1.mp4' 
    players_json = r'C:\Users\Yingbo.Jiao22\Desktop\surf\output\players1.json' 
    balls_json = r'C:\Users\Yingbo.Jiao22\Desktop\surf\output\basketball1.json'  
    output_path = r'C:\Users\Yingbo.Jiao22\Desktop\surf\output\visualized.mp4' 

    visualize(video_path, players_json, balls_json, output_path)