from PIL import Image
from ultralytics import YOLO
import glob
import os
# Load a pretrained YOLOv8n model
model = YOLO('../ultralytics/runs/detect/train32/weights/best.pt')

# Run inference on 'bus.jpg'
image_path = '/home/serena/Data/Birds/lesser_frigates/LF_yolov8_50/split_data/test/images/*.jpg'
out_folder ='/home/serena/Data/Birds/lesser_frigates/LF_yolov8_50/test_results/' 
image_files = sorted(glob.glob(image_path))
for im in image_files:
    results = model(im)  # results list
    im_out = im.split('/')[-1] 
    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        #input()
        im.save(os.path.join(out_folder, im_out))  # save image
