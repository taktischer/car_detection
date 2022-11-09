from yolov5 import YOLOv5
import os

# set model params
model_path = "yolov5/weights/yolov5s.pt"
device = "cpu"

yolov5 = YOLOv5(model_path, device)

for file in os.listdir('./pictures'):
    if file.endswith(('png', 'jpg', 'jpeg')):
        results = yolov5.predict(f'./pictures/{file}')
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        print(results)
        results.show()
