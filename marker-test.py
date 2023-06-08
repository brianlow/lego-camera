import os
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image, ImageDraw

from lib.lego_colors import lego_colors_by_id

dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
detector = aruco.ArucoDetector(dict)

def symbol_value_to_color_id(symbol_value):
    if symbol_value < 50:
        return symbol_value
    else:
        return (symbol_value-50) * 50

filenames = [
    'tmp/markers-1.jpeg',
    'tmp/markers-2.jpeg',
    'tmp/markers-3.jpeg',
    'tmp/markers-4.jpeg',
    'tmp/markers-5.jpeg',
    'tmp/markers-6.jpeg',
    'tmp/markers-7.jpeg',
    'tmp/markers-8.jpeg',
    'tmp/markers-9.jpeg',
    'tmp/markers-10.jpeg',
    'tmp/markers-11.jpeg',
    'tmp/markers-12.jpeg',
    'tmp/markers-13.jpeg',
    'tmp/markers-14.jpeg',
    'tmp/markers-15.jpeg',
    'tmp/markers-16.jpeg',
]
for filename in filenames:
    print(f"Detecting {filename}...")
    img = cv2.imread(filename)

    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    ids = sorted([list[0] for list in ids])
    print(f"Found {len(corners)} with ids {ids}")

    # the corner symbols are value 99
    color_ids = [id for id in ids if id != 99]
    if len(color_ids) == 2:
        print(f"color_ids: {color_ids}")
        color_id = symbol_value_to_color_id(color_ids[0])
        color_id += symbol_value_to_color_id(color_ids[1])
        color = lego_colors_by_id[color_id]
        print(f"Found {color.id} - {color.name}")

    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    for corner in corners:
        points = [tuple(point) for point in corner[0].astype(int)]  # convert corners to integer
        draw.line((*points[0], *points[1], *points[2], *points[3], *points[0]), fill='red', width=3)

    name, ext = os.path.splitext(filename)
    new_filename = f"{name}-out{ext}"
    img.save(new_filename)
