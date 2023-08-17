from ultralytics import YOLO

model = YOLO('weights/yolov8x-cls.pt')

model.train(data='../../Data/Turtles/classifier',
            pretrained=False,
            epochs=100,
            imgsz=64
            )
