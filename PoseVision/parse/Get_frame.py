import cv2

video_path = r"C:\Users\Yingbo.Jiao\Desktop\mmpose\input\sample.mp4"
cap = cv2.VideoCapture(video_path)

success, frame = cap.read()
if success:
    cv2.imwrite(r"C:\Users\Yingbo.Jiao\Desktop\mmpose\input\reference_frame.jpg", frame)
    print("参考帧已保存为 reference_frame.jpg")
else:
    print("无法读取视频帧")

cap.release()