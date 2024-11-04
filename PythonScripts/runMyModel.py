import ultralytics
from ultralytics import YOLO

model = YOLO('D:\Raw dataset\yoloModel.tflite')

results = model.predict('D:\Raw dataset\Viable\Img00041.png', save=True, imgsz=640, conf=.5)

num_detections = len(results[0].boxes)

print(f'The number of detected seeds is : {num_detections}')