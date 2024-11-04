import ultralytics
from ultralytics import YOLO

model = YOLO('D:\Raw dataset\model\Corn_types_float32.tflite')

result = model.predict('D:\Raw dataset\combined.MP4', save=True, imgsz=640, conf=.55)

#num_detections = len(result[0].boxes)

#print(f'Number of detected seeds : {num_detections}')