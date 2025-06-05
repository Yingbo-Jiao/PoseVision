import cv2
from yolo_detection. import 
from filter. import 
from tracking. import 
from utils import 

video_path = "video.mp4"
detector = 
tracker 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
  

cap.release()
cv2.destroyAllWindows()
