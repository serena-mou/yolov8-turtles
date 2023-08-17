from ultralytics import YOLO
# load pretrained model
model = YOLO('runs/detect/train17/weights/best.pt')
# train the model
model.track(source='../../Data/Turtles/videos/081217-00020AMsouth.mp4',
            save=True,
            imgsz=640,
            show=True
            )
print('done')
