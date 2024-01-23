from ultralytics import YOLO
# load pretrained model
# model = YOLO('weights/yolov8x.pt')
model = YOLO('weights/yolov8n.pt')
# train the model
model.train(data='data/chondria.yaml',
            pretrained=True,
            epochs=400,
            batch=4,
            imgsz=640,
            close_mosaic=0,
            classes = [0]
            )
print('done')
