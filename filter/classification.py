#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import glob
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
import random
import torch
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import os
import sys

# 添加 YOLO 檢測器路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolo_detection')))
from wrapper import YOLODetector

def arg_parse():
    """
    Parse arguments 
    """
    parser = argparse.ArgumentParser(description='player identification and team recognition')
    parser.add_argument("--videos", dest='videos', required=True,
                      help="video / Directory containing videos to perform detection upon", type=str)
    return parser.parse_args()

class TeamColorDrawer:
    """
    A class to handle team color drawing based on player detection
    """
    def __init__(self):
        # 明確定義深藍色和白色隊伍的顏色
        self.team_colors = {
            0: (139, 0, 0),    # 深藍色 (DarkBlue) - BGR格式
            1: (255, 255, 255) # 白色 (White) - BGR格式
        }
        self.team_names = {
            0: "DarkBlue",
            1: "White"
        }
    
    def draw_players(self, frame, detections, team_assignments):
        """
        Draw players with team colors on the frame
        """
        for det, team_id in zip(detections, team_assignments):
            x1, y1, x2, y2, conf, class_id = det
            if class_id == 0:  # Only process players (class_id=0)
                color = self.team_colors.get(team_id, (0, 0, 255))  # 默認紅色表示錯誤
                team_name = self.team_names.get(team_id, "Unknown")
                
                # 繪製邊框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 繪製隊伍標籤和置信度
                label = f"{team_name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

class CLAHEPreprocessor:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessor
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=tile_grid_size)
    
    def preprocess(self, img):
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge channels back
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

class TemporalSmoother:
    """
    Temporal smoothing for team assignments
    """
    def __init__(self, alpha=0.7, max_history=5):
        self.alpha = alpha  # Weight for previous frame's assignment
        self.max_history = max_history  # Max frames to remember per player
        self.history = defaultdict(list)  # Track player assignment history
    
    def update_assignments(self, detections, current_assignments):
        """
        Apply temporal smoothing to current assignments
        """
        smoothed_assignments = []
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = det
            player_key = f"{x1:.1f}_{y1:.1f}_{x2:.1f}_{y2:.1f}"  # Simple key based on position
            
            # Get current assignment
            current_assignment = current_assignments[i]
            
            # Update history for this player
            self.history[player_key].append(current_assignment)
            if len(self.history[player_key]) > self.max_history:
                self.history[player_key].pop(0)
            
            # Calculate smoothed assignment
            if len(self.history[player_key]) > 1:
                # Weighted average of current and historical assignments
                hist_assignments = np.array(self.history[player_key][:-1])
                smoothed = (self.alpha * np.mean(hist_assignments) + 
                           (1 - self.alpha) * current_assignment)
                smoothed_assignment = int(round(smoothed))
            else:
                smoothed_assignment = current_assignment
            
            smoothed_assignments.append(smoothed_assignment)
        
        return smoothed_assignments

def extract_features_with_model(imgs, model, layer_name="conv2d_15"):
    """
    Extract features from images using a model
    """
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = intermediate_layer_model.predict(imgs)
    return features.reshape(features.shape[0], -1)

def determine_team_clusters(centers):
    """
    根據聚類中心確定哪個是深藍色隊伍，哪個是白色隊伍
    """
    # 計算每個聚類的藍色通道強度
    blue_strength = centers[:, 0]  # BGR格式中的B通道
    
    # 找出最藍的聚類 (深藍色隊伍)
    blue_team_idx = np.argmax(blue_strength)
    
    # 找出最白的聚類 (白色隊伍)
    white_team_idx = np.argmin(np.std(centers, axis=1))  # 顏色最均勻的
    
    # 確保兩個隊伍不同
    if blue_team_idx == white_team_idx:
        # 如果相同，選擇第二藍的作為藍隊
        sorted_indices = np.argsort(blue_strength)[::-1]
        blue_team_idx = sorted_indices[0]
        white_team_idx = sorted_indices[1]
    
    return blue_team_idx, white_team_idx

def main(video_path):
    # 初始化組件
    args = arg_parse()
    yolo_detector = YOLODetector()
    team_drawer = TeamColorDrawer()
    clahe = CLAHEPreprocessor()
    smoother = TemporalSmoother()
    
    # 加載autoencoder模型用於特徵提取
    json_file = open('C:/Users/子衿/Desktop/cv/PoseVision/project-RG/models/convautoencodermodel_10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:/Users/子衿/Desktop/cv/PoseVision/project-RG/models/convautoencodermodel_10.h5")
    
    # 訓練KMeans模型
    print("[INFO] 正在訓練KMeans模型進行隊伍識別...")
    capture = cv2.VideoCapture(video_path)
    sample_frames = []
    for _ in range(100):  # 採樣100幀用於訓練
        ret, frame = capture.read()
        if not ret:
            break
        sample_frames.append(frame)
    capture.release()
    
    # 處理採樣幀獲取球員圖像
    player_imgs = []
    for frame in sample_frames:
        # 應用CLAHE預處理
        frame = clahe.preprocess(frame)
        
        detections = yolo_detector.detect_image(frame)
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            if class_id == 0:  # 只處理球員
                crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop_img.size > 0:
                    try:
                        img_resized = cv2.resize(crop_img, (64, 64))
                        player_imgs.append(img_resized)
                    except:
                        continue
    
    if len(player_imgs) == 0:
        print("[錯誤] 在採樣幀中未檢測到球員！")
        return
    
    # 提取特徵並訓練KMeans
    player_imgs_np = np.array(player_imgs)
    features = extract_features_with_model(player_imgs_np, loaded_model)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)  # 只分2個集群
    
    # 確定哪個集群對應哪個隊伍
    centers = kmeans.cluster_centers_
    blue_team_idx, white_team_idx = determine_team_clusters(centers)
    
    print(f"[INFO] 隊伍分配結果: 深藍色隊伍={blue_team_idx}, 白色隊伍={white_team_idx}")
    
    # 處理整個視頻
    print("[INFO] 開始視頻處理...")
    capture = cv2.VideoCapture(video_path)
    frames = []
    width, height = None, None
    
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
            
        if width is None or height is None:
            height, width = frame.shape[:2]
        
        # 應用CLAHE預處理
        frame_processed = clahe.preprocess(frame.copy())
        
        # 檢測球員
        detections = yolo_detector.detect_image(frame_processed)
        player_detections = [det for det in detections if det[5] == 0]  # 只處理球員
        
        if len(player_detections) > 0:
            # 提取球員圖像
            player_imgs = []
            valid_detections = []
            for det in player_detections:
                x1, y1, x2, y2, conf, class_id = det
                crop_img = frame_processed[int(y1):int(y2), int(x1):int(x2)]
                if crop_img.size > 0:
                    try:
                        img_resized = cv2.resize(crop_img, (64, 64))
                        player_imgs.append(img_resized)
                        valid_detections.append(det)
                    except:
                        continue
            
            if len(player_imgs) > 0:
                # 預測隊伍分配
                player_imgs_np = np.array(player_imgs)
                features = extract_features_with_model(player_imgs_np, loaded_model)
                labels = kmeans.predict(features)
                
                # 映射標籤到隊伍ID
                team_assignments = []
                for label in labels:
                    if label == blue_team_idx:
                        team_assignments.append(0)  # 深藍色隊伍
                    else:
                        team_assignments.append(1)  # 白色隊伍
                
                # 應用時序平滑
                smoothed_assignments = smoother.update_assignments(valid_detections, team_assignments)
                
                # 用隊伍顏色繪製球員
                frame = team_drawer.draw_players(frame, valid_detections, smoothed_assignments)
        
        frames.append(frame)
    
    capture.release()
    
    # 保存輸出視頻
    print("[INFO] 正在保存結果...")
    size = (width, height)
    out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for frame in frames:
        out.write(frame)
    out.release()
    print("[INFO] 處理完成！")

if __name__ == "__main__":
    args = arg_parse()
    main(args.videos)