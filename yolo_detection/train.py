from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt')  
    model.train(
        data=r"C:\Users\Yingbo.Jiao\Desktop\PoseVision\configs\data.yaml",  
        epochs=100,
        imgsz=2560,
        batch=16,
        device=0
    )

if __name__ == " _main_":
    main

