from ultralytics import YOLO
# load pretrained model
model = YOLO('runs/detect/train17/weights/best.pt')
# train the model
model.predict(source='../../Data/Turtles/videos/081217-00020AMsouth.mp4',
            save=True,
            imgsz=640,
            save_txt=True,
            save_conf=True
            )
print('done')
