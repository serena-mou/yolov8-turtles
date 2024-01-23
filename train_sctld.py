from ultralytics import YOLO
# load pretrained model
# model = YOLO('weights/yolov8x.pt')
model = YOLO('weights/yolov8n.pt')
# train the model
model.train(data='data/sctld-init.yaml',
            pretrained=True,
            epochs=400,
            batch=4,
            imgsz=320,
            patience=0
            )
print('done')
