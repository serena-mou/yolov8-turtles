from ultralytics import YOLO
# load pretrained model
#model = YOLO('weights/yolov8x.pt')
model = YOLO('weights/yolov8m.pt')
# train the model
model.train(data='data/LF.yaml',
            pretrained=True,
            lr0=0.01,
            epochs=100,
            imgsz=640,
            workers=4,
            amp=False,
            batch=-1
            )
print('done')
