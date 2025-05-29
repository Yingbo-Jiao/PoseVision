import cv2
from parse_labelme import parse_labelme

# 读取图像
image = cv2.imread('reference_frame.jpg')

# 解析 JSON 获取点坐标
points = parse_labelme(r"C:\Users\Yingbo.Jiao\Desktop\mmpose\parse\reference_frame.json")
roi_points = tuple(item for sublist in points for item in sublist)

print(roi_points)
