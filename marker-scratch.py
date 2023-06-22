import os
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from lib.lego_colors import lego_colors_by_id
from lib.aruco_utils import aruco_ids_to_color_id
from lib.bounding_box import BoundingBox
from lib.predictor import Predictor

dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
detector = aruco.ArucoDetector(dict)
detection_model = YOLO("detect-10-4k-real-and-renders-nano-1024-image-size2.pt")
predictor = Predictor(detection_model, None, None, None)


filenames = [
    'tmp/last-capture-original.jpeg',
]
for filename in filenames:
    print(f"Detecting {filename}...")
    image = Image.open(filename)

    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    aruco_boxes, ids, rejectedImgPoints = detector.detectMarkers(opencv_image)

    draw = ImageDraw.Draw(image)

    for corner in aruco_boxes:
        points = [tuple(point) for point in corner[0].astype(int)]  # convert corners to integer
        draw.line((*points[0], *points[1], *points[2], *points[3], *points[0]), fill='red', width=5)

    # Find the corner symbols. These are the symbols with value 99.
    # Since we are scanning 3d boxes with the symbol on all sides, we combine
    # any symbols that are close together.
    boxes = []
    for index, id in enumerate(ids):
        if id == 99:
            boxes.append(BoundingBox.from_aruco(aruco_boxes[index][0]))
    boxes = BoundingBox.combine_nearby(boxes, threshold=0.1*image.width)
    for box in boxes:
        box.grow(10).draw(draw, color='yellow', width=5)

    cropping_box = boxes[0].combine(boxes[1]).combine(boxes[2]).combine(boxes[3])
    cropping_box.grow(20).draw(draw, color='blue', width=5)

    cropped_image = cropping_box.crop(image)
    pieces = predictor.detect_objects(cropped_image)
    print(f"Found {len(pieces)} lego pieces")

    for piece in pieces:
        piece.move(cropping_box.x1, cropping_box.y1).draw(draw, color='purple', width=5)

    # ids = sorted([list[0] for list in ids])
    # print(f"Found {len(corners)} with ids {ids}")

    # # the corner symbols are value 99
    # color_ids = [id for id in ids if id != 99]
    # if len(color_ids) == 2:
    #     print(f"color_ids: {color_ids}")
    #     color_id = aruco_ids_to_color_id(color_ids)
    #     color = lego_colors_by_id[color_id]
    #     print(f"Found {color.id} - {color.name}")
    #     draw.line((*points[0], *points[1], *points[2], *points[3], *points[0]), fill='red', width=3)

    name, ext = os.path.splitext(filename)
    new_filename = f"{name}-out{ext}"
    image.save(new_filename)

print("done")
