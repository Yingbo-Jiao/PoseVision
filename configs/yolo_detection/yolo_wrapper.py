from ultralytics import YOLO
import numpy as np
import cv2
import json

class YOLODetector:
    def __init__(self, model_path='yolo_detection/best.pt', conf_threshold=0.25):
        """
        Initialize the YOLO detector.

        Parameters:
        - model_path (str): Path to the trained YOLO model.
        - conf_threshold (float): Minimum confidence threshold to keep detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = {
            0: 'player',
            1: 'ball',
            2: 'referee',
            3: 'other'
        }

    def detect_image(self, image):
        """
        Perform object detection on a single image.

        Parameters:
        - image (np.ndarray): Input image in BGR format.

        Returns:
        - detections (List[List[float]]): A list of [x1, y1, x2, y2, confidence, class_id].
        """
        results = self.model(image)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            if conf >= self.conf_threshold:
                detections.append([x1, y1, x2, y2, conf, class_id])
        return detections

    def detect_video(self, video_path, output_video_path=None, json_output_path=None):
        """
        Perform detection on every frame of a video.
        Save annotated video and detection results as JSON.

        Parameters:
        - video_path (str): Path to input video.
        - output_video_path (str): Path to save annotated video.
        - json_output_path (str): Path to save detection results as JSON.

        Returns:
        - frame_results (List[List[List[float]]]): List of detection results for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        frame_results = []
        frame_idx = 0

        # Set up video writer if needed
        writer = None
        if output_video_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_image(frame)
            frame_results.append(detections)

            # Draw detections
            for x1, y1, x2, y2, conf, cls_id in detections:
                label = f"{self.class_names.get(cls_id, str(cls_id))} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer:
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        if json_output_path:
            self.save_to_json(frame_results, json_output_path)

        return frame_results

    def save_to_json(self, frame_results, json_path):
        """
        Save detection results to a JSON file.

        Parameters:
        - frame_results (List[List[List[float]]]): Detection results for each frame.
        - json_path (str): Path to save the JSON file.
        """
        json_data = {}
        for i, detections in enumerate(frame_results):
            json_data[f'frame_{i+1}'] = [
                {
                    'x1': round(det[0], 2),
                    'y1': round(det[1], 2),
                    'x2': round(det[2], 2),
                    'y2': round(det[3], 2),
                    'confidence': round(det[4], 4),
                    'class_id': int(det[5]),
                    'class_name': self.class_names.get(int(det[5]), str(det[5]))
                }
                for det in detections
            ]
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
