import numpy as np
import cv2
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.models import Model
from collections import defaultdict
from yolo_detection.detector import YOLODetector


class CLAHEPreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def preprocess(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


class TemporalSmoother:
    def __init__(self, alpha=0.7, max_history=5):
        self.alpha = alpha
        self.max_history = max_history
        self.history = defaultdict(list)

    def update_assignments(self, detections, current_assignments):
        smoothed_assignments = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            key = f"{x1:.1f}_{y1:.1f}_{x2:.1f}_{y2:.1f}"
            curr = current_assignments[i]
            self.history[key].append(curr)
            if len(self.history[key]) > self.max_history:
                self.history[key].pop(0)
            if len(self.history[key]) > 1:
                hist = np.array(self.history[key][:-1])
                smooth = self.alpha * np.mean(hist) + (1 - self.alpha) * curr
                smoothed_assignments.append(int(round(smooth)))
            else:
                smoothed_assignments.append(curr)
        return smoothed_assignments


def extract_features_with_model(imgs, model, layer_name="conv2d_15"):
    imgs = imgs.astype('float32') / 255.0
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = intermediate_model.predict(imgs)
    return features.reshape(features.shape[0], -1)


def determine_team_clusters(centers):
    blue_strength = centers[:, 0]
    blue_team_idx = np.argmax(blue_strength)
    white_team_idx = np.argmin(np.std(centers, axis=1))
    if blue_team_idx == white_team_idx:
        sorted_idx = np.argsort(blue_strength)[::-1]
        blue_team_idx = sorted_idx[0]
        white_team_idx = sorted_idx[1]
    return blue_team_idx, white_team_idx


class TeamClassifier:
    def __init__(self, autoencoder_json_path, autoencoder_weights_path, sample_video_path=None):
        self.clahe = CLAHEPreprocessor()
        self.smoother = TemporalSmoother()
        self.model = self._load_autoencoder(autoencoder_json_path, autoencoder_weights_path)
        self.kmeans = None
        self.blue_team_idx = None
        self.white_team_idx = None
        if sample_video_path is not None:
            self._train_kmeans(sample_video_path)

    def _load_autoencoder(self, json_path, weights_path):
        with open(json_path, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model

    def _train_kmeans(self, video_path):
        print("[INFO] 正在從樣本幀訓練 KMeans...")
        capture = cv2.VideoCapture(video_path)
        sample_frames = []
        for _ in range(100):
            ret, frame = capture.read()
            if not ret:
                break
            sample_frames.append(frame)
        capture.release()

        player_imgs = []
        yolo = YOLODetector()
        for frame in sample_frames:
            frame = self.clahe.preprocess(frame)
            detections = yolo.detect_image(frame)
            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                if class_id == 0:
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        try:
                            img = cv2.resize(crop, (64, 64))
                            player_imgs.append(img)
                        except:
                            continue

        if not player_imgs:
            raise ValueError("無法從樣本中提取球員圖像")

        features = extract_features_with_model(np.array(player_imgs), self.model)
        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        self.blue_team_idx, self.white_team_idx = determine_team_clusters(self.kmeans.cluster_centers_)
        print(f"[INFO] KMeans 訓練完成: 深藍隊={self.blue_team_idx}, 白隊={self.white_team_idx}")

    def classify_frame(self, frame, detections):
        frame = self.clahe.preprocess(frame)
        player_imgs = []
        valid_dets = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                try:
                    img = cv2.resize(crop, (64, 64))
                    player_imgs.append(img)
                    valid_dets.append(det)
                except:
                    continue

        if not player_imgs:
            return [dict(det, team_id=-1) for det in detections]

        features = extract_features_with_model(np.array(player_imgs), self.model)
        raw_labels = self.kmeans.predict(features)
        team_assignments = [0 if l == self.blue_team_idx else 1 for l in raw_labels]
        smoothed = self.smoother.update_assignments(valid_dets, team_assignments)

        output_dets = []
        idx = 0
        for det in detections:
            if det['class_id'] == 0:
                det['team_id'] = smoothed[idx]
                idx += 1
            else:
                det['team_id'] = -1
            output_dets.append(det)

        return output_dets
