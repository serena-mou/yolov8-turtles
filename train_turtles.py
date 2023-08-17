from ultralytics import YOLO
# load pretrained model
#model = YOLO('weights/yolov8x.pt')
model = YOLO('runs/detect/train21/weights/last.pt')
# train the model
model.train(data='data/turtles.yaml',
            pretrained=True,
            lr0=0.05,
            epochs=100,
            imgsz=640,
            workers=4,
            amp=False,
            batch=-1
            )
print('done')
