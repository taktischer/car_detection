from yolov5 import YOLOv5
from configparser import ConfigParser
from PIL import Image

# yolov5 definitions
model_path = "yolov5/weights/yolov5s.pt"
device = "cpu"
yolov5 = YOLOv5(model_path, device)

# load config file
config = ConfigParser()
config.read('config.ini')
free_parking_lots = int(config['DEFAULT']['parking_lots'])


# calculate the free parking lots
# requires a list of tensors from yolov5
def get_free_parking_lots(prediction) -> int:
    global free_parking_lots
    area = 0

    for idx, obj in enumerate(prediction):
        width, height = int(obj[2]) - int(obj[0]), int(obj[3]) - int(obj[1])

        # define x0/y0 for the base rectangle
        # define x1/y1 for the other rectangle
        # uses the previous area to check which one is the base rectangle
        if area > width * height:
            x0, x1, y0, y1 = int(prediction[idx - 1][0]), int(obj[0]), int(prediction[idx - 1][1]), int(obj[1])
        else:
            x0, x1, y0, y1 = int(obj[0]), int(prediction[idx - 1][0]), int(obj[1]), int(prediction[idx - 1][1])

        # change the area to the new area
        area = width * height

        # checks if the percentage in the base rectangle is below 25%
        if (x0 - x1) / width * 100 > 25.0 or (y0 - y1) / width * 100 > 25.0:
            free_parking_lots -= 1
    return free_parking_lots


# get the prediction value from yolov5
# image_path, camera_position are required
def predict_image(camera_position: int):
    # opening picture and bitmap
    image_0 = Image.open(fr"pictures\parkinglots\parkinglot_{camera_position}.png")  # NOQA
    bitmap = Image.open(fr"pictures\bitmaps\bitmap_{camera_position}.png")

    # create new image with size of the picture
    new_im = Image.new('RGBA', (image_0.size[0], image_0.size[1]), (250, 250, 250))

    # paste the bitmap on top
    new_im.paste(image_0, (0, 0), mask=bitmap)

    # modify image_path and save the new image in results folder
    new_im.save(fr".\pictures\results\result_{camera_position}.png")

    # get prediction from yolov5
    results = yolov5.predict(fr".\pictures\results\result_{camera_position}.png")

    # return the list of tensor values
    return results.pred[0]


if __name__ == "__main__":
    print(get_free_parking_lots(predict_image(3)))
