from ultralytics import YOLO
# load pretrained model
#model = YOLO('weights/yolov8x.pt')
model = YOLO('weights/yolov8m.pt')
# train the model
model.train(data='data/LF.yaml',
            pretrained=True,
            epochs=100,
            imgsz=416,
            workers=4,
            batch=-1
            )
print('done')
