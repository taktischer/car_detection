from yolov5 import YOLOv5
from PIL import Image

# yolov5 definitions
model_path = "yolov5/weights/yolov5s.pt"
device = "cpu"
yolov5 = YOLOv5(model_path, device)


# calculate the free parking lots
# requires a list of tensors from yolov5
def get_occupied_parking_lots(prediction, camera_position) -> list[int]:
    area = 0
    occupied_parking_spaces = []

    # opens the overlay image, to check which parking spaces are occupied
    image = Image.open(fr".\pictures\bitmaps\bitmap_{camera_position}_overlay.png")
    # image.show()
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
            # get the rgba value (overlay image) at the left bottom corner of the rectangle
            # if the value of red isn't 0 it will be added to the occupied_parking_spaces
            # the red value is the parking space id
            position = image.getpixel((int(obj[0]), int(obj[3])))[0]
            if position != 0:
                occupied_parking_spaces.append(position)

    image.close()
    return occupied_parking_spaces


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

    # close the image after modification
    new_im.close()

    # get prediction from yolov5
    results = yolov5.predict(fr".\pictures\results\result_{camera_position}.png")

    # return the list of tensor values
    return results.pred[0]


if __name__ == "__main__":
    camera = 3
    predicted_image = predict_image(camera)
    occupied_parking_lots: list = get_occupied_parking_lots(predicted_image, camera)
