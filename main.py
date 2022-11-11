from yolov5 import YOLOv5
from configparser import ConfigParser

model_path = "yolov5/weights/yolov5s.pt"
device = "cpu"

yolov5 = YOLOv5(model_path, device)

config = ConfigParser()
config.read('config.ini')
free_parking_lots = int(config['DEFAULT']['parking_lots'])


def get_parking_lots(image_path: str) -> int:
    global free_parking_lots
    results = yolov5.predict(image_path)
    for category in results.pred[0][:, 5]:
        if int(category) in [2]:  # cars category: 2;
            free_parking_lots -= 1
    return free_parking_lots


print(get_parking_lots('./pictures/picture_1.jpeg'))
