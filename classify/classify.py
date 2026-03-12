# Legacy experimental team classification script.
# The reusable project interface lives in classify/team_classifier.py.

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import random
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from sklearn.cluster import MiniBatchKMeans
import os
import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

clf   = LogisticRegression(max_iter=1000)
scaler = StandardScaler()

# 训练阶段：一次性冷启动（可选，也可直接空模型）
# clf.fit(X_train, y_train)   # 如果没有人工起始数据，可跳过

# 添加 YOLO 檢測器路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolo_detection')))
from detector import YOLODetector

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

def extract_features_with_model(imgs, model, layer_name="conv2d_15"):
    """
    Extract features from images using a model
    """
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = intermediate_layer_model.predict(imgs)
    return features.reshape(features.shape[0], -1)

def determine_team_clusters(kmeans, player_imgs):
    # 获取每个聚类对应的原始图像
    labels = kmeans.labels_
    cluster_0_imgs = [player_imgs[i] for i, l in enumerate(labels) if l == 0]
    cluster_1_imgs = [player_imgs[i] for i, l in enumerate(labels) if l == 1]

    # 计算每个聚类的平均BGR颜色
    mean_color_0 = np.mean(cluster_0_imgs, axis=(0, 1, 2))
    mean_color_1 = np.mean(cluster_1_imgs, axis=(0, 1, 2))

    print("Cluster 0 平均BGR:", mean_color_0)
    print("Cluster 1 平均BGR:", mean_color_1)

    # 根据B通道判断哪个更蓝
    blue_idx = 1 if mean_color_0[0] > mean_color_1[0] else 0
    white_idx = 1 - blue_idx  # 另一个聚类为白色队伍
    return blue_idx, white_idx

class TeamMainRunner:
    """
    仅封装：把你提供的 main(video_path) 一字不改地搬进来作为类方法 run(video_path)。
    算法、流程、人工校验交互（39~40秒区间、按键切换/删除/回车确认、保存 X,y 并监督学习）
    全部保持不变。
    """
    def run(self, video_path):
        # ===== 你的 main(video_path) 内容开始（保持原样） =====
        # 初始化组件
        # args = arg_parse()  # 封装后直接传入 video_path，这行不再需要解析命令行
        yolo_detector = YOLODetector()
        team_drawer = TeamColorDrawer()
        clahe = CLAHEPreprocessor()
        
        # 加载autoencoder模型用于特征提取
        json_file = open(r'C:\Users\Yingbo.Jiao\Desktop\PoseVision\classify\project-RG\models\convautoencodermodel_10.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(r'C:\Users\Yingbo.Jiao\Desktop\PoseVision\classify\project-RG\models\convautoencodermodel_10.h5')
        
        # 用视频训练KMeans
        print("[INFO] 用demo.mp4训练KMeans队伍识别...")
        train_video = r"C:\Users\Yingbo.Jiao\Desktop\sample1.mp4"
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        capture = cv2.VideoCapture(train_video)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        num_samples = min(350, total_frames)
        num_segments = 10
        samples_per_segment = num_samples // num_segments if num_segments > 0 else num_samples
        sample_indices = []
        for i in range(num_segments):
            start = i * (total_frames // num_segments)
            end = (i + 1) * (total_frames // num_segments)
            end = min(end, total_frames)
            if end > start:
                segment_range = list(range(start, end))
                if len(segment_range) < samples_per_segment:
                    segment_samples = segment_range
                else:
                    segment_samples = random.sample(segment_range, samples_per_segment)
                sample_indices.extend(segment_samples)
        if len(sample_indices) < num_samples:
            remaining = set(range(total_frames)) - set(sample_indices)
            sample_indices.extend(random.sample(list(remaining), num_samples - len(sample_indices)))
        sample_indices = sorted(sample_indices)
        sample_frames = []
        for idx in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = capture.read()
            if not ret:
                continue
            sample_frames.append(frame)
        capture.release()

        # 只统计非黑色像素的球员图像
        player_imgs = []
        blue_channel_means = []
        for frame in sample_frames:
            frame = clahe.preprocess(frame)
            detections = yolo_detector.detect_image(frame)

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                if class_id == 0 and conf > 0.7:  # 只处理置信度高的球员
                    crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop_img.size > 0:
                        # 只保留非黑色像素
                        mask = np.any(crop_img > 10, axis=2)
                        crop_img_nobg = crop_img.copy()
                        crop_img_nobg[~mask] = 0
                        try:
                            img_resized = cv2.resize(crop_img_nobg, (64, 64))
                            player_imgs.append(img_resized)
                            blue_channel_means.append(np.mean(img_resized[:,:,0]))  # 存储蓝色通道均值
                        except:
                            continue
        
        if len(player_imgs) == 0:
            print("[错误] 在采样帧中未检测到球员！")
            return
        
        # 提取特征并训练KMeans
        player_imgs_np = np.array(player_imgs)
        features = extract_features_with_model(player_imgs_np, loaded_model)
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=0)
        kmeans.partial_fit(features)  # 初始化在线KMeans模型

        pca = PCA(n_components=2).fit_transform(features)
        plt.scatter(pca[:,0], pca[:,1], c=kmeans.labels_, cmap='coolwarm'); 
        plt.show()
        
        # 确定哪个集群对应哪个队伍
        blue_team_idx, white_team_idx = determine_team_clusters(kmeans, player_imgs)
        
        print(f"[INFO] 队伍分配结果: 深蓝色队伍={blue_team_idx}, 白色队伍={white_team_idx}")

        capture = cv2.VideoCapture("demo.mp4")
        original_fps = int(capture.get(cv2.CAP_PROP_FPS))
        start_frame = int(35 * original_fps)
        end_frame = int(40 * original_fps)
        
        frame_count = 0
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            frame_count += 1
            if frame_count < start_frame or frame_count > end_frame:
                continue

            frame_processed = clahe.preprocess(frame.copy())
            detections = yolo_detector.detect_image(frame_processed)
            
            # 只处理置信度高的球员
            player_detections = [det for det in detections if det[5] == 0 and det[4] > 0.7]
            

            if len(player_detections) > 0:
                player_imgs = []
                valid_detections = []
                for det in player_detections:
                    x1, y1, x2, y2, conf, class_id = det
                    crop_img = frame_processed[int(y1):int(y2), int(x1):int(x2)]
                    if crop_img.size > 0:
                        mask = np.any(crop_img > 10, axis=2)
                        crop_img_nobg = crop_img.copy()
                        crop_img_nobg[~mask] = 0
                        try:
                            img_resized = cv2.resize(crop_img_nobg, (64, 64))
                            blue_channel_means.append(np.mean(img_resized[:,:,0]))  # 第0个通道是蓝色通道
                            player_imgs.append(img_resized)
                            valid_detections.append(det)  # 存储元组
                        except:
                            continue
            
            if len(player_imgs) > 0:
                player_imgs_np = np.array(player_imgs)
                features = extract_features_with_model(player_imgs_np, loaded_model)
                kmeans.partial_fit(features)  # 更新在线KMeans模型
                labels = kmeans.predict(features)
                team_assignments = []
                for label in labels:
                    if label == blue_team_idx:
                        team_assignments.append(0)
                    else:
                        team_assignments.append(1)

                vis = frame_processed.copy()
                for i, det in enumerate(valid_detections):
                    x1, y1, x2, y2 = list(map(int, det[:4]))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), team_drawer.team_colors[team_assignments[i]], 2)
                    cv2.putText(vis, f"{i}", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_drawer.team_colors[team_assignments[i]], 2)
                cv2.putText(vis, f"Frame:{frame_count:04d}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("frame", vis)

                    

                # ---------- 非阻塞键盘交互 ----------
                print("\n【提示】0=DarkBlue  1=White")
                for idx, (det) in enumerate(valid_detections):
                    print(f"[{idx}] {det[:4]} 当前队伍 {team_assignments[idx]}")

                # 在主循环中初始化索引列表
            indices = list(range(len(valid_detections)))

            # ---- 全局 ----
            final_X, final_y = [], []
            manual_log = []  # 用于记录人工确认的框和队伍分配
            
            while True:
                key = cv2.waitKey(0) & 0xFF

                # —— 回车：下一帧 + 压栈 ——
                if key == 13:
                    for det, label in zip(valid_detections, team_assignments):
                        x1, y1, x2, y2, _, _ = det
                        crop = frame_processed[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size == 0:
                            continue
                        mask = np.any(crop > 10, axis=2)
                        crop_nobg = crop.copy()
                        crop_nobg[~mask] = 0
                        img = cv2.resize(crop_nobg, (64, 64))
                        feat = extract_features_with_model(np.array([img]), loaded_model)[0]
                        final_X.append(feat)
                        final_y.append(label)
                    manual_log.append({
                        "frame_idx": frame_count,
                        "bboxes": [list(map(float, d)) for d in valid_detections],
                        "teams": team_assignments.copy()  })
                    break

                # —— 数字键改队伍 ——
                elif ord('0') <= key <= ord('9'):
                    idx = key - ord('0')
                    if 0 <= idx < len(valid_detections):
                        new_team = 1 - team_assignments[idx]
                        team_assignments[idx] = new_team
                        # 立即重画
                        frame_copy = frame.copy()
                        for i, (d) in enumerate(valid_detections):
                            x, y, X, Y = list(map(int, d[:4]))
                            t = team_assignments[i]
                            col = team_drawer.team_colors[t]
                            cv2.rectangle(frame_copy, (x, y), (X, Y), col, 2)
                            cv2.putText(frame_copy, f"{i}", (x, y-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                        cv2.imshow("frame", frame_copy)
                        
                # —— d 键：人工删除误检框 ——
                elif key == ord('d'):
                    sub_key = cv2.waitKey(0) & 0xFF
                    idx = sub_key - ord('0')
                    if 0 <= idx < len(valid_detections):
                        del valid_detections[idx]
                        del team_assignments[idx]
                        indices.pop(idx)  # 更新索引列表
                        # 重画
                        frame_copy = frame.copy()
                        for i, det in enumerate(valid_detections):
                            x, y, X, Y = list(map(int, det[:4]))
                            t = team_assignments[i]
                            col = team_drawer.team_colors[t]
                            cv2.rectangle(frame_copy, (x, y), (X, Y), col, 2)
                            cv2.putText(frame_copy, f"{i}", (x, y-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                        cv2.imshow("frame", frame_copy)
        capture.release()

        # 保存
        joblib.dump((final_X, final_y), 'manual_labels.pkl')

        # 加载
        final_X, final_y = joblib.load('manual_labels.pkl')
        def collect_extra_frames(video_path, num_frames=150):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = random.sample(range(total), num_frames)
            extra_X, extra_y = [], []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_processed = clahe.preprocess(frame)
                detections = yolo_detector.detect_image(frame_processed)
                player_boxes = [d for d in detections if d[5] == 0 and d[4] > 0.7]

                for det in player_boxes:
                    x1, y1, x2, y2, _, _ = det
                    crop = frame_processed[int(y1):int(y2), int(x1):int(x2)]
                    crop = cv2.resize(crop, (64, 64))
                    feat = extract_features_with_model(np.array([crop]), loaded_model)[0]
                    extra_X.append(feat)
                    # 用当前 KMeans 打标签（或人工快速确认）
                    feature=extract_features_with_model(np.array([crop]), loaded_model)
                    labels = kmeans.predict(feature)
                    team_assignments = []
                    for label in labels:
                        if label == blue_team_idx:
                            team_assignments.append(0)
                        else:
                            team_assignments.append(1)
                    extra_y.append(team_assignments[0])
            cap.release()
            return extra_X, extra_y
        # 在 main() 最后，把额外 150 帧加入训练
        extra_X, extra_y = collect_extra_frames("demo.mp4", 200)
        final_X.extend(extra_X)
        final_y.extend(extra_y)
        if final_X:
            clf = LogisticRegression(max_iter=1000)
            scaler = StandardScaler()
            clf.fit(scaler.fit_transform(np.array(final_X)), np.array(final_y))
            joblib.dump((clf, scaler), 'final_clf.pkl')
        print("[INFO] 处理完成！")

        # 加载人工模型
        clf, scaler = joblib.load('final_clf.pkl')

        # 打开待标注视频
        in_path  = "demo.mp4"
        out_path = "demo_labeled.mp4"
        cap = cv2.VideoCapture(in_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        clahe = CLAHEPreprocessor()
        team_drawer = TeamColorDrawer()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_processed = clahe.preprocess(frame)
            detections = yolo_detector.detect_image(frame_processed)

            # 筛选球员
            player_boxes = [d for d in detections if d[5] == 0 and d[4] > 0.7]
            imgs = []

            for x1, y1, x2, y2, _, _ in player_boxes:
                crop = frame_processed[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                mask = np.any(crop > 10, axis=2)
                crop_nobg = crop.copy()
                crop_nobg[~mask] = 0
                try:
                    img = cv2.resize(crop_nobg, (64, 64))
                    imgs.append(img)
                except:
                    continue

            if imgs:
                feats = extract_features_with_model(np.array(imgs), loaded_model)
                preds = clf.predict(scaler.transform(feats))
            else:
                preds = []

            # 画框
            for i, (det, team) in enumerate(zip(player_boxes, preds)):
                x1, y1, x2, y2 = list(map(int, det[:4]))
                color = team_drawer.team_colors[team]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                print(frame, (x1, y1), (x2, y2),team_drawer.team_names[team])
                cv2.putText(frame, str(i), (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        out.release()
        print("✅ 带标签视频已保存到", out_path)
        # ===== 你的 main(video_path) 内容结束 =====


if __name__ == "__main__":
    runner = TeamMainRunner()
    runner.run(video_path="demo.mp4")
