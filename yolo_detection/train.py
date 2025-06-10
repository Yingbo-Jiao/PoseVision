from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

# Train
model.train(
    data=r"C:\Users\Yingbo.Jiao\Desktop\PoseVision\configs\data.yaml",  
    epochs=100,
    imgsz=2560,
    batch=16,
    workers=4,
    device=0
)
    

