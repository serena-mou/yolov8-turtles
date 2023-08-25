from ultralytics import YOLO

model = YOLO('weights/yolov8s-cls.pt')

model.train(data='../../Data/Turtles/classifier',
            pretrained=True,
            epochs=100,
            imgsz=64
            )
