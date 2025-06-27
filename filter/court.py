from shapely.geometry import Point, Polygon

def filter_detections_by_roi(detections, roi_points):
    """
    过滤检测框,只保留下边框中点在多边形ROI内的目标

    参数:
        detections (dict): 目标检测结果列表,每个dict含 'bbox' 键，格式为[x1, y1, x2, y2]
        roi_points (tuples): ROI四边形顶点列表 [(x1, y1), (x2, y2), ...]

    返回:
        dict: 过滤后有效的目标列表，如果没有返回空列表
    """
    roi_polygon = Polygon(roi_points)
    valid_detections = []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center_bottom = Point((x1 + x2) / 2, y2)  

        if roi_polygon.contains(center_bottom):
            valid_detections.append(det)

    return valid_detections

import json
import random
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from pathlib import Path

class Court:
    def __init__(self, reference_image_path=None, saved_polygon_path=None):
        '''
        Initialize the Court class.

        Parameters:
        - reference_image_path (str): Path to an image used to define the court area.
        - saved_polygon_path (str): Optional path to an existing saved polygon file (JSON format).
        '''
        self.reference_image_path = reference_image_path
        self.saved_polygon_path = saved_polygon_path or 'court_polygon.json'
        self.polygon = self.load_or_select_polygon()

    def extract_random_frame_from_video(self, video_path, output_path='court_frame.jpg'):
        """
        Extract a random frame from the input video and save it as an image.

        Parameters:
        - video_path (str): Path to the input video file.
        - output_path (str): Where to save the extracted frame image.

        Returns:
        - str: Path to the saved frame image.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Video has no frames.")

        frame_number = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video.")

        cv2.imwrite(output_path, frame)
        cap.release()

        return output_path
    
    def load_or_select_polygon(self):
        '''
        Load an existing court polygon from file or allow manual selection if not found.

        Returns:
        - polygon (Polygon): A Shapely Polygon object representing the court.
        '''
        if Path(self.saved_polygon_path).exists():
            with open(self.saved_polygon_path, 'r') as f:
                points = json.load(f)
            print(f"[INFO] Loaded existing court polygon from {self.saved_polygon_path}")
        else:
            print("[INFO] No existing court polygon found. Please select manually.")
            image = cv2.imread(self.reference_image_path)
            points = []

            def click_event(event, x, y, flags, param):
                # Capture mouse click coordinates and draw circle for feedback
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Select 4 court corners", image)

            # Display image and let user click to select polygon corners
            cv2.imshow("Select 4 court corners", image)
            cv2.setMouseCallback("Select 4 court corners", click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(points) != 4:
                raise ValueError("You must select exactly 4 points to define the court area.")

            # Save selected points to JSON file
            with open(self.saved_polygon_path, 'w') as f:
                json.dump(points, f)
            print(f"[INFO] Saved court polygon to {self.saved_polygon_path}")

        return Polygon(points)

    def filter_detections_by_polygon(self, detections, image_shape):
        '''
        Filter YOLO detections based on whether they fall inside the defined court polygon.

        Parameters:
        - detections (List[List[float]]): List of detection results in format [x1, y1, x2, y2, conf, class_id]
        - image_shape (Tuple[int, int]): Shape of the image (height, width)

        Returns:
        - filtered_detections (List[List[float]]): Detections inside the polygon or all 'ball' detections.
        '''
        height, width = image_shape[:2]
        filtered_detections = []

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            # Always retain 'ball' class (class_id == 1) regardless of polygon
            if int(class_id) == 1:
                filtered_detections.append(det)
                continue

            # Calculate bottom center point of the bounding box (more robust than using bbox center)
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            point = Point(bottom_center_x, bottom_center_y)

            # Include detection only if this point lies within the court polygon
            if self.polygon.contains(point):
                filtered_detections.append(det)

        return filtered_detections
