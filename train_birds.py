from ultralytics import YOLO
# load pretrained model
# model = YOLO('weights/yolov8x.pt')
model = YOLO('weights/yolov8m.pt')
# train the model
model.train(data='data/MBay.yaml',
            pretrained=True,
            epochs=400,
            imgsz=512,
            batch=4,
            classes=[0,1,2,3,4,5,6]
            )
print('done')
