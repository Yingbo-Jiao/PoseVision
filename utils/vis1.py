import cv2
import json
import os
import numpy as np
from tqdm import tqdm

def visualize_results(video_path, 
                    classified_json_path,
                    main_json_path,
                    output_video_path):
    """
    可视化处理流程：
    1. 读取分类结果绘制队伍边界框
    2. 读取主结果绘制骨骼关键点
    3. 输出最终视频
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 加载JSON数据
    with open(classified_json_path) as f:
        classified_data = json.load(f)
    with open(main_json_path) as f:
        main_data = json.load(f)
    
    # COCO-17关键点连接关系
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
        [2, 4], [3, 5], [4, 6], [5, 7]]
    
    # 颜色定义
    COLORS = {
        'team0': (255, 255, 255),  # 白色 (队伍A)
        'team1': (255, 165, 0),    # 蓝色 (队伍B)
        'ball': (0, 0, 255),       # 红色 (球)
        'skeleton': (0, 255, 0)    # 绿色 (骨骼)
    }
    
    for frame_idx in tqdm(range(total_frames), desc="可视化处理"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_key = f"frame_{frame_idx+1}"
        
        # ===== 第一步：绘制分类结果 =====
        if frame_key in classified_data:
            # 绘制球员边界框
            for player in classified_data[frame_key].get('classified_players', []):
                team_id = player.get('team_id', 0)
                color = COLORS[f'team{team_id}']
                x1, y1, x2, y2 = map(int, player['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 显示team_id
                cv2.putText(frame, f"Team {team_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制球边界框
            for ball in classified_data[frame_key].get('ball_detections', []):
                x1, y1, x2, y2 = map(int, ball['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['ball'], 2)
                cv2.putText(frame, "Ball", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['ball'], 1)
        
        # ===== 第二步：绘制主结果 =====
        if frame_key in main_data:
            # 绘制骨骼关键点
            for player in main_data[frame_key].get('players', []):
                pose = np.array(player.get('pose', []))
                
                # 绘制关键点
                for point in pose:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 3, COLORS['skeleton'], -1)
                
                # 绘制骨骼连接线
                for i, j in SKELETON:
                    if i-1 < len(pose) and j-1 < len(pose):
                        start = tuple(map(int, pose[i-1]))
                        end = tuple(map(int, pose[j-1]))
                        cv2.line(frame, start, end, COLORS['skeleton'], 2)
                
                # 显示跟踪ID
                if 'id' in player and 'bbox' in player:
                    x1, y1, x2, y2 = map(int, player['bbox'])
                    cv2.putText(frame, f"ID: {player['id']}", (x1, y1-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # 写入帧
        out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    print(f"[SUCCESS] 可视化视频已保存至: {output_video_path}")

if __name__ == "__main__":
    # 使用示例
    video_path = r"C:\Users\Yingbo.Jiao22\Desktop\PoseVision\input\sample1.mp4"
    classified_json = r"C:\Users\Yingbo.Jiao22\Desktop\PoseVision\classified_results.json"
    main_json = r"C:\Users\Yingbo.Jiao22\Desktop\PoseVision\outputanalysis_results.json"
    output_video = r"C:\Users\Yingbo.Jiao22\Desktop\PoseVision\output/visualization.mp4"
    
    visualize_results(
        video_path=video_path,
        classified_json_path=classified_json,
        main_json_path=main_json,
        output_video_path=output_video
    )